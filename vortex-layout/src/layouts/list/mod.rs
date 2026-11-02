// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An experimental structural layout for list-typed columns.

mod expr;
mod reader;
pub mod writer;

use std::sync::Arc;

use reader::ListReader;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Layout;
use crate::LayoutChildType;
use crate::LayoutDeserializeArgs;
use crate::LayoutId;
use crate::LayoutParts;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::OwnedLayoutChildren;
use crate::segments::SegmentSource;

/// Child index of the elements layout.
pub const ELEMENTS_CHILD_INDEX: usize = 0;
/// Child index of the offsets layout.
pub const OFFSETS_CHILD_INDEX: usize = 1;
/// Child index of the optional validity layout.
pub const VALIDITY_CHILD_INDEX: usize = 2;
/// Number of children for a non-nullable list.
pub const NUM_CHILDREN_NON_NULLABLE: usize = 2;

/// List layout vtable.
#[derive(Clone, Debug)]
pub struct List;

/// Backwards-compatible name for the list layout plugin.
pub use List as ListLayoutEncoding;

/// List-layout-specific data.
#[derive(Clone, Debug)]
pub struct ListData {
    offsets_ptype: PType,
}

/// A list layout shredded into elements, offsets, and optional validity children.
pub type ListLayout = Layout<List>;

impl VTable for List {
    type LayoutData = ListData;
    type Metadata = ProstMetadata<ListLayoutMetadata>;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.list");
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
        ProstMetadata(ListLayoutMetadata::new(layout.offsets_ptype))
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &ListLayoutMetadata,
    ) -> VortexResult<Self::LayoutData> {
        let offsets_ptype = metadata.get_offsets_ptype()?;
        ListLayout::validate_children(args.dtype, args.children.nchildren())?;
        let elements_dtype = args
            .dtype
            .as_list_element_opt()
            .ok_or_else(|| vortex_err!("ListLayout requires a List dtype, got {}", args.dtype))?;
        args.children.child(ELEMENTS_CHILD_INDEX, elements_dtype)?;
        let offsets_dtype = DType::Primitive(offsets_ptype, Nullability::NonNullable);
        let offsets = args.children.child(OFFSETS_CHILD_INDEX, &offsets_dtype)?;
        let offsets_rows = args.row_count.checked_add(1).ok_or_else(|| {
            vortex_err!(
                "List row count {} overflows its offsets row count",
                args.row_count
            )
        })?;
        vortex_ensure_eq!(
            offsets.row_count(),
            offsets_rows,
            "List offsets row count must be the parent row count plus one"
        );
        if args.dtype.is_nullable() {
            let validity = args
                .children
                .child(VALIDITY_CHILD_INDEX, &DType::Bool(Nullability::NonNullable))?;
            vortex_ensure_eq!(
                validity.row_count(),
                args.row_count,
                "List validity row count does not match parent"
            );
        }
        Ok(ListData { offsets_ptype })
    }

    fn nslots(_layout: &Layout<Self>) -> usize {
        // Elements, offsets, and an always-slotted (optionally present) validity child.
        VALIDITY_CHILD_INDEX + 1
    }

    fn slot_to_child(layout: &Layout<Self>, slot: usize) -> Option<usize> {
        match slot {
            ELEMENTS_CHILD_INDEX | OFFSETS_CHILD_INDEX => Some(slot),
            VALIDITY_CHILD_INDEX => layout.dtype().is_nullable().then_some(VALIDITY_CHILD_INDEX),
            _ => None,
        }
    }

    fn child_dtype(layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
        match idx {
            ELEMENTS_CHILD_INDEX => layout
                .dtype()
                .as_list_element_opt()
                .map(|dtype| dtype.as_ref().clone())
                .ok_or_else(|| vortex_err!("ListLayout requires a List dtype")),
            OFFSETS_CHILD_INDEX => Ok(DType::Primitive(
                layout.offsets_ptype,
                Nullability::NonNullable,
            )),
            VALIDITY_CHILD_INDEX if layout.dtype().is_nullable() => {
                Ok(DType::Bool(Nullability::NonNullable))
            }
            _ => vortex_bail!("Invalid child index {idx} for ListLayout"),
        }
    }

    fn child_type(layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        match idx {
            ELEMENTS_CHILD_INDEX => LayoutChildType::Auxiliary("elements".into()),
            OFFSETS_CHILD_INDEX => LayoutChildType::Auxiliary("offsets".into()),
            VALIDITY_CHILD_INDEX if layout.dtype().is_nullable() => {
                LayoutChildType::Auxiliary("validity".into())
            }
            _ => vortex_panic!("Invalid child index {idx} for ListLayout"),
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ListReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
            ctx,
        )?))
    }
}

impl Layout<List> {
    /// Construct a list layout from its children, checking them the way deserialization does: the
    /// elements match the list's element dtype, the offsets are non-nullable integers with one
    /// more row than the list, and a nullable list has a non-nullable bool validity child with one
    /// row per list.
    pub fn try_new(
        dtype: DType,
        elements: LayoutRef,
        offsets: LayoutRef,
        validity: Option<LayoutRef>,
    ) -> VortexResult<Self> {
        let elements_dtype = dtype
            .as_list_element_opt()
            .ok_or_else(|| vortex_err!("ListLayout requires a List dtype, got {dtype}"))?;
        vortex_ensure_eq!(
            elements.dtype(),
            elements_dtype.as_ref(),
            "ListLayout elements dtype does not match the list's element dtype"
        );
        let offsets_ptype = match offsets.dtype() {
            DType::Primitive(ptype, Nullability::NonNullable) if ptype.is_int() => *ptype,
            other => vortex_bail!("ListLayout offsets must be non-nullable integers, got {other}"),
        };
        let row_count = offsets
            .row_count()
            .checked_sub(1)
            .ok_or_else(|| vortex_err!("ListLayout offsets must have at least one row"))?;
        if let Some(validity) = &validity {
            vortex_ensure_eq!(
                validity.dtype(),
                &DType::Bool(Nullability::NonNullable),
                "ListLayout validity must be non-nullable bool"
            );
            vortex_ensure_eq!(
                validity.row_count(),
                row_count,
                "List validity row count does not match parent"
            );
        }
        let mut children = vec![elements, offsets];
        children.extend(validity);
        Self::validate_children(&dtype, children.len())?;
        Ok(LayoutParts::new(
            List,
            dtype,
            row_count,
            Vec::new(),
            OwnedLayoutChildren::layout_children(children),
            ListData { offsets_ptype },
        )
        .into_typed())
    }

    /// Returns the elements child.
    pub fn elements(&self) -> VortexResult<LayoutRef> {
        self.slot(ELEMENTS_CHILD_INDEX)?
            .ok_or_else(|| vortex_err!("ListLayout elements slot is absent"))
    }

    /// Returns the offsets child.
    pub fn offsets(&self) -> VortexResult<LayoutRef> {
        self.slot(OFFSETS_CHILD_INDEX)?
            .ok_or_else(|| vortex_err!("ListLayout offsets slot is absent"))
    }

    /// Returns the optional validity child.
    pub fn validity(&self) -> VortexResult<Option<LayoutRef>> {
        self.slot(VALIDITY_CHILD_INDEX)
    }

    /// Returns the integer ptype used by offsets.
    pub fn offsets_ptype(&self) -> PType {
        self.offsets_ptype
    }

    /// Returns the list element dtype.
    pub fn elements_dtype(&self) -> &DType {
        self.dtype()
            .as_list_element_opt()
            .vortex_expect("ListLayout dtype must be a List")
    }

    fn validate_children(dtype: &DType, nchildren: usize) -> VortexResult<()> {
        let expected = NUM_CHILDREN_NON_NULLABLE + usize::from(dtype.is_nullable());
        vortex_ensure_eq!(nchildren, expected);
        Ok(())
    }
}

#[derive(prost::Message)]
pub struct ListLayoutMetadata {
    #[prost(int32, tag = "1")]
    offsets_ptype: i32,
}

impl ListLayoutMetadata {
    pub fn new(offsets_ptype: PType) -> Self {
        Self {
            offsets_ptype: offsets_ptype as i32,
        }
    }

    fn get_offsets_ptype(&self) -> VortexResult<PType> {
        let ptype = PType::try_from(self.offsets_ptype).map_err(|_| {
            vortex_err!(
                "ListLayoutMetadata offsets_ptype must be a known PType, got {}",
                self.offsets_ptype
            )
        })?;
        vortex_ensure!(
            ptype.is_int(),
            "ListLayoutMetadata offsets_ptype must be an integer PType, got {ptype}"
        );
        Ok(ptype)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use prost::Message;
    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;
    use vortex_session::registry::ReadContext;

    use super::List;
    use super::ListLayout;
    use super::ListLayoutMetadata;
    use crate::LayoutBuildContext;
    use crate::LayoutDeserialization;
    use crate::LayoutPlugin;
    use crate::LayoutRef;
    use crate::layout_children;
    use crate::layouts::flat::FlatLayout;
    use crate::test::SESSION;

    /// Deserialize a non-nullable `list<i32>` node whose offsets child has `offsets_rows` rows.
    fn deserialize(ptype: PType, row_count: u64, offsets_rows: u64) -> VortexResult<LayoutRef> {
        let elements = DType::Primitive(PType::I32, Nullability::NonNullable);
        let dtype = DType::List(Arc::new(elements.clone()), Nullability::NonNullable);
        let offsets = DType::Primitive(ptype, Nullability::NonNullable);
        let read_ctx = ReadContext::new([]);
        let children = layout_children(vec![
            FlatLayout::new(0, elements, 0.into(), read_ctx.clone()).into_layout(),
            FlatLayout::new(offsets_rows, offsets, 1.into(), read_ctx.clone()).into_layout(),
        ]);
        let metadata = ListLayoutMetadata::new(ptype).encode_to_vec();
        LayoutPlugin::deserialize(
            &List,
            LayoutDeserialization {
                serialized_id: LayoutPlugin::id(&List),
                dtype: &dtype,
                row_count,
                metadata: &metadata,
                segment_ids: vec![],
                children: children.as_ref(),
            },
            &LayoutBuildContext {
                session: &SESSION,
                array_read_ctx: &read_ctx,
            },
        )
    }

    #[rstest]
    #[case::float_offsets(PType::F64, 3, 4, "must be an integer PType, got f64")]
    #[case::zero_rows_without_offsets(PType::U64, 0, 0, "parent row count plus one")]
    #[case::extra_offsets(PType::U64, 3, 5, "parent row count plus one")]
    #[case::row_count_overflow(PType::U64, u64::MAX, u64::MAX, "overflows")]
    fn deserialize_rejects_malformed_offsets(
        #[case] ptype: PType,
        #[case] row_count: u64,
        #[case] offsets_rows: u64,
        #[case] message: &str,
    ) {
        let err = deserialize(ptype, row_count, offsets_rows)
            .err()
            .map(|err| err.to_string());
        assert!(
            err.as_ref().is_some_and(|err| err.contains(message)),
            "{err:?}"
        );
    }

    #[rstest]
    #[case(PType::U64, 0, 1)]
    #[case(PType::U32, 3, 4)]
    #[case(PType::I64, 3, 4)]
    fn deserialize_accepts_one_offset_per_row_plus_one(
        #[case] ptype: PType,
        #[case] row_count: u64,
        #[case] offsets_rows: u64,
    ) -> VortexResult<()> {
        assert_eq!(
            deserialize(ptype, row_count, offsets_rows)?.row_count(),
            row_count
        );
        Ok(())
    }

    /// `ListLayout::try_new` rejects children that deserialization would reject.
    #[rstest]
    #[case::elements_dtype(
        PType::I64,
        PType::U64,
        Nullability::NonNullable,
        4,
        None,
        "elements dtype"
    )]
    #[case::float_offsets(PType::I32, PType::F64, Nullability::NonNullable, 4, None, "integers")]
    #[case::nullable_offsets(PType::I32, PType::U64, Nullability::Nullable, 4, None, "integers")]
    #[case::no_offsets(
        PType::I32,
        PType::U64,
        Nullability::NonNullable,
        0,
        None,
        "at least one row"
    )]
    #[case::validity_rows(
        PType::I32,
        PType::U64,
        Nullability::NonNullable,
        4,
        Some(2),
        "validity row count"
    )]
    fn try_new_rejects_mismatched_children(
        #[case] elements_ptype: PType,
        #[case] offsets_ptype: PType,
        #[case] offsets_nullability: Nullability,
        #[case] offsets_rows: u64,
        #[case] validity_rows: Option<u64>,
        #[case] message: &str,
    ) {
        let read_ctx = ReadContext::new([]);
        let list_nullability = Nullability::from(validity_rows.is_some());
        let dtype = DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            list_nullability,
        );
        let flat = |rows, dtype, segment: u32| {
            FlatLayout::new(rows, dtype, segment.into(), read_ctx.clone()).into_layout()
        };
        let elements = flat(
            0,
            DType::Primitive(elements_ptype, Nullability::NonNullable),
            0,
        );
        let offsets = flat(
            offsets_rows,
            DType::Primitive(offsets_ptype, offsets_nullability),
            1,
        );
        let validity =
            validity_rows.map(|rows| flat(rows, DType::Bool(Nullability::NonNullable), 2));
        let err = ListLayout::try_new(dtype, elements, offsets, validity)
            .err()
            .map(|err| err.to_string());
        assert!(
            err.as_ref().is_some_and(|err| err.contains(message)),
            "{err:?}"
        );
    }

    #[test]
    fn test_build_rejects_unknown_offsets_ptype() -> VortexResult<()> {
        let dtype = DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::NonNullable,
        );
        let metadata = ListLayoutMetadata { offsets_ptype: 11 }.encode_to_vec();
        let children = layout_children(vec![]);
        let read_ctx = ReadContext::new([]);
        let build_ctx = LayoutBuildContext {
            session: &SESSION,
            array_read_ctx: &read_ctx,
        };
        let result = LayoutPlugin::deserialize(
            &List,
            LayoutDeserialization {
                serialized_id: LayoutPlugin::id(&List),
                dtype: &dtype,
                row_count: 0,
                metadata: &metadata,
                segment_ids: vec![],
                children: children.as_ref(),
            },
            &build_ctx,
        );
        assert!(result.is_err_and(|err| {
            err.to_string()
                .contains("ListLayoutMetadata offsets_ptype must be a known PType, got 11")
        }));

        assert_eq!(
            ListLayoutMetadata::new(PType::U32).get_offsets_ptype()?,
            PType::U32
        );

        Ok(())
    }
}
