// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Golden files pinning the `vortex.list` layout that `vortex.listLayout` writes.
//!
//! The list layout is in our frozen `qc2026.11.0` edition, which promises its files stay readable.
//! Upstream makes no such promise: it keeps the layout in its unstable edition. Each file in
//! `goldenfiles/list_layout/` was written once and is never rewritten:
//!
//! - every file must still read back to the values it was written from, through full scans, row
//!   ranges, row indices, projections, filters and list pushdown;
//! - today's writer must still write the same list layouts: their metadata, children, dtypes and
//!   row counts.
//!
//! A change that fails either check changes the format, and must not be merged as is. To keep
//! reading the old files, the reader keeps the old format under its wire id. Don't regenerate a
//! file to make a test pass; see `goldenfiles/list_layout/README.md`.

use std::fs;
use std::path::PathBuf;

use rstest::rstest;
use rstest_reuse::apply;
use rstest_reuse::template;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::ListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::stream::ArrayStreamExt;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::expr::Lambda;
use vortex::expr::col;
use vortex::expr::get_item;
use vortex::expr::gt_eq;
use vortex::expr::list_contains;
use vortex::expr::list_transform;
use vortex::expr::lit;
use vortex::expr::root;
use vortex::expr::select;
use vortex::expr::var;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::LayoutChildType;
use vortex::layout::LayoutRef;
use vortex::layout::VTable;
use vortex::layout::layouts::list::List;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::mask::Mask;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex::session::VortexSession;
use vortex_array::assert_arrays_eq;

use crate::RUNTIME;
use crate::data_source::file_open_options;
use crate::session::new_session;
use crate::writer::write_session;

/// Rows of every non-empty case: more than two of the writer's 8192-row blocks, so lists and
/// their elements are split across chunks.
const ROWS: usize = 20_000;

/// The cases, each a struct of an `id` column (its row index) and list columns.
#[derive(Clone, Copy, Debug)]
enum Case {
    /// Non-nullable lists of non-nullable `i64`, some empty.
    Primitive,
    /// Nullable lists of nullable `i32`, with a long run of null lists.
    Nullable,
    /// Lists of nullable strings, both inlined and too long to inline.
    Utf8,
    /// Lists of structs.
    StructElements,
    /// Nullable lists of nullable lists.
    Nested,
    /// A list column inside a struct column.
    InStruct,
    /// Lists that are all empty.
    AllEmpty,
}

impl Case {
    fn name(self) -> &'static str {
        match self {
            Case::Primitive => "primitive",
            Case::Nullable => "nullable",
            Case::Utf8 => "utf8",
            Case::StructElements => "struct_elements",
            Case::Nested => "nested",
            Case::InStruct => "in_struct",
            Case::AllEmpty => "all_empty",
        }
    }

    fn path(self) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("goldenfiles/list_layout")
            .join(format!("{}.vortex", self.name()))
    }

    /// The values the case's file was written from.
    fn array(self, session: &VortexSession) -> VortexResult<ArrayRef> {
        let rows = ROWS;
        let id = Buffer::from_iter(0..i64::try_from(ROWS)?).into_array();
        let lists = match self {
            Case::Primitive => {
                let offsets = offsets(rows, |row| row % 6);
                let elements = Buffer::from_iter((0..last(&offsets)).map(|i| i64::from(i) * 3));
                list(
                    session,
                    elements.into_array(),
                    offsets,
                    Validity::NonNullable,
                )?
            }
            Case::Nullable => {
                let offsets = offsets(rows, |row| (row * 7 + 3) % 5);
                let elements = PrimitiveArray::from_option_iter(
                    (0..last(&offsets)).map(|i| (i % 5 != 0).then_some(i32::try_from(i).ok()?)),
                );
                let validity = Validity::from_iter(
                    (0..rows).map(|row| row % 11 != 0 && !(5_000..6_000).contains(&row)),
                );
                list(session, elements.into_array(), offsets, validity)?
            }
            Case::Utf8 => {
                let offsets = offsets(rows, |row| row % 4);
                let elements =
                    VarBinViewArray::from_iter_nullable_str((0..last(&offsets)).map(|i| {
                        (i % 7 != 0).then(|| {
                            if i % 3 == 0 {
                                format!("a string too long to be inlined in a view: {i}")
                            } else {
                                format!("s{i}")
                            }
                        })
                    }));
                list(
                    session,
                    elements.into_array(),
                    offsets,
                    Validity::NonNullable,
                )?
            }
            Case::StructElements => {
                let offsets = offsets(rows, |row| row % 5);
                let n = last(&offsets);
                let elements = StructArray::from_fields(&[
                    ("a", Buffer::from_iter((0..n).map(i64::from)).into_array()),
                    (
                        "b",
                        VarBinViewArray::from_iter_nullable_str(
                            (0..n).map(|i| (i % 4 != 0).then(|| format!("b{i}"))),
                        )
                        .into_array(),
                    ),
                ])?;
                list(
                    session,
                    elements.into_array(),
                    offsets,
                    Validity::NonNullable,
                )?
            }
            Case::Nested => {
                let outer = offsets(rows, |row| row % 3);
                let inner_rows = usize::try_from(last(&outer))?;
                let inner = offsets(inner_rows, |row| (row * 5 + 1) % 4);
                let elements = PrimitiveArray::from_option_iter(
                    (0..last(&inner)).map(|i| (i % 6 != 0).then_some(i32::try_from(i).ok()?)),
                );
                let inner_validity = Validity::from_iter((0..inner_rows).map(|row| row % 9 != 0));
                let inner = list(session, elements.into_array(), inner, inner_validity)?;
                let outer_validity = Validity::from_iter((0..rows).map(|row| row % 13 != 0));
                list(session, inner, outer, outer_validity)?
            }
            Case::InStruct => {
                let offsets = offsets(rows, |row| (row * 3) % 4);
                let elements = Buffer::from_iter((0..last(&offsets)).map(|i| f64::from(i) / 4.0));
                let l = list(
                    session,
                    elements.into_array(),
                    offsets,
                    Validity::NonNullable,
                )?;
                StructArray::from_fields(&[("l", l)])?.into_array()
            }
            Case::AllEmpty => list(
                session,
                Buffer::<i64>::empty().into_array(),
                offsets(rows, |_| 0),
                Validity::NonNullable,
            )?,
        };
        let name = if matches!(self, Case::InStruct) {
            "s"
        } else {
            "l"
        };
        Ok(StructArray::from_fields(&[("id", id), (name, lists)])?.into_array())
    }
}

/// Offsets of `rows` lists, the list of row `row` holding `len(row)` elements.
fn offsets(rows: usize, len: impl Fn(usize) -> usize) -> Buffer<u32> {
    let mut end = 0u32;
    std::iter::once(0)
        .chain((0..rows).map(|row| {
            end += u32::try_from(len(row)).vortex_expect("list length fits u32");
            end
        }))
        .collect()
}

fn last(offsets: &Buffer<u32>) -> u32 {
    offsets.last().copied().unwrap_or(0)
}

fn list(
    session: &VortexSession,
    elements: ArrayRef,
    offsets: Buffer<u32>,
    validity: Validity,
) -> VortexResult<ArrayRef> {
    Ok(ListArray::try_new(
        elements,
        offsets.into_array(),
        validity,
        &mut session.create_execution_ctx(),
    )?
    .into_array())
}

/// Writes `array` as a `vortex.listLayout` write does.
fn write(array: ArrayRef) -> VortexResult<ByteBuffer> {
    let session = write_session(&new_session(), true)?;
    let mut bytes = ByteBufferMut::empty();
    RUNTIME.block_on(
        session
            .write_options()
            .write(&mut bytes, array.to_array_stream()),
    )?;
    Ok(bytes.freeze())
}

/// Opens `bytes` as the JNI reader does.
fn open(session: &VortexSession, bytes: ByteBuffer) -> VortexResult<VortexFile> {
    file_open_options(session.open_options()).open_buffer(bytes)
}

fn golden(session: &VortexSession, case: Case) -> VortexResult<VortexFile> {
    let path = case.path();
    let bytes = fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{} is missing ({e}); see goldenfiles/list_layout/README.md",
            path.display()
        )
    });
    open(session, ByteBuffer::from(bytes))
}

/// Every list layout in `layout`'s tree, one line each: its path, dtype, rows, metadata, and
/// each child's name, dtype and rows. Paths name only struct fields and list children, so
/// changes to the layouts around a list, such as its zones or chunks, don't show here.
fn list_layouts(layout: &LayoutRef) -> VortexResult<String> {
    let mut out = String::new();
    describe(layout, "$", &mut out)?;
    Ok(out)
}

fn describe(layout: &LayoutRef, path: &str, out: &mut String) -> VortexResult<()> {
    let children = layout.children()?;
    let names: Vec<_> = layout.child_names().collect();
    let is_list = layout.encoding_id() == List.id();
    if is_list {
        let metadata: String = layout
            .metadata()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        out.push_str(&format!(
            "{path}: {} rows={} metadata={metadata}",
            layout.dtype(),
            layout.row_count()
        ));
        for (name, child) in names.iter().zip(&children) {
            out.push_str(&format!(" {name}={}x{}", child.dtype(), child.row_count()));
        }
        out.push('\n');
    }
    for ((name, kind), child) in names.iter().zip(layout.child_types()).zip(&children) {
        let named = is_list || matches!(kind, LayoutChildType::Field(_));
        let child_path = if named {
            format!("{path}/{name}")
        } else {
            path.to_string()
        };
        describe(child, &child_path, out)?;
    }
    Ok(())
}

fn scan(
    file: &VortexFile,
    configure: impl FnOnce(ScanBuilder<ArrayRef>) -> VortexResult<ScanBuilder<ArrayRef>>,
) -> VortexResult<ArrayRef> {
    let builder = configure(file.scan()?)?;
    RUNTIME.block_on(builder.into_array_stream()?.read_all())
}

fn bind(file: &VortexFile, expr: &Expression) -> VortexResult<vortex::expr::BoundExpression> {
    expr.bind(file.dtype())?.optimize_recursive()
}

/// The rows of `array` that `filter` keeps, evaluated in memory.
fn filtered(
    session: &VortexSession,
    array: &ArrayRef,
    filter: &Expression,
) -> VortexResult<ArrayRef> {
    let keep = array
        .clone()
        .apply(filter)?
        .execute::<BoolArray>(&mut session.create_execution_ctx())?;
    array.filter(Mask::from_iter(keep.into_bit_buffer().iter()))
}

const CASES: [Case; 7] = [
    Case::Primitive,
    Case::Nullable,
    Case::Utf8,
    Case::StructElements,
    Case::Nested,
    Case::InStruct,
    Case::AllEmpty,
];

#[template]
#[rstest]
#[case::primitive(Case::Primitive)]
#[case::nullable(Case::Nullable)]
#[case::utf8(Case::Utf8)]
#[case::struct_elements(Case::StructElements)]
#[case::nested(Case::Nested)]
#[case::in_struct(Case::InStruct)]
#[case::all_empty(Case::AllEmpty)]
fn cases(#[case] case: Case) {}

#[apply(cases)]
fn golden_file_has_list_layouts(#[case] case: Case) -> VortexResult<()> {
    let session = new_session();
    let layouts = list_layouts(golden(&session, case)?.footer().layout())?;
    assert!(
        !layouts.is_empty(),
        "{} has no vortex.list layout",
        case.name()
    );
    Ok(())
}

#[apply(cases)]
fn writer_writes_the_golden_list_layouts(#[case] case: Case) -> VortexResult<()> {
    let session = new_session();
    let golden = list_layouts(golden(&session, case)?.footer().layout())?;
    let written = open(&session, write(case.array(&session)?)?)?;
    assert_eq!(
        list_layouts(written.footer().layout())?,
        golden,
        "the list layouts written for {} changed: files written now differ from the golden \
         file, which readers of the old format may not read",
        case.name()
    );
    Ok(())
}

#[apply(cases)]
fn golden_file_reads_whole(#[case] case: Case) -> VortexResult<()> {
    let session = new_session();
    let file = golden(&session, case)?;
    let actual = scan(&file, Ok)?;
    assert_arrays_eq!(
        actual,
        case.array(&session)?,
        &mut session.create_execution_ctx()
    );
    Ok(())
}

/// A row range across the first block boundary, and one that ends the file.
#[apply(cases)]
fn golden_file_reads_row_ranges(
    #[case] case: Case,
    #[values(8_000..8_400, 19_990..20_000)] range: std::ops::Range<u64>,
) -> VortexResult<()> {
    let session = new_session();
    let file = golden(&session, case)?;
    let actual = scan(&file, |scan| Ok(scan.with_row_range(range.clone())))?;
    let expected = case
        .array(&session)?
        .slice(usize::try_from(range.start)?..usize::try_from(range.end)?)?;
    assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
    Ok(())
}

#[apply(cases)]
fn golden_file_reads_row_indices(#[case] case: Case) -> VortexResult<()> {
    let session = new_session();
    let file = golden(&session, case)?;
    let indices = [0u64, 1, 4_999, 5_000, 8_191, 8_192, 8_193, 16_384, 19_999];
    let actual = scan(&file, |scan| {
        Ok(
            scan.with_row_indices(StrictSortedBuffer::try_new(Buffer::from_iter(
                indices.iter().copied(),
            ))?),
        )
    })?;
    let expected = case
        .array(&session)?
        .take(Buffer::from_iter(indices.iter().copied()).into_array())?;
    assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
    Ok(())
}

/// Each projection, read from the file and evaluated on the values in memory.
#[rstest]
#[case::list_column(Case::Primitive, select(["l"], root()))]
#[case::nullable_list_column(Case::Nullable, col("l"))]
#[case::nested_list_column(Case::Nested, col("l"))]
#[case::list_in_a_struct(Case::InStruct, get_item("l", col("s")))]
#[case::element_field(Case::StructElements, list_transform(col("l"), element("a")))]
#[case::nullable_element_field(Case::StructElements, list_transform(col("l"), element("b")))]
#[case::utf8_elements(Case::Utf8, col("l"))]
fn golden_file_reads_projections(
    #[case] case: Case,
    #[case] projection: Expression,
) -> VortexResult<()> {
    let session = new_session();
    let file = golden(&session, case)?;
    let actual = scan(&file, |scan| {
        Ok(scan.with_projection(bind(&file, &projection)?))
    })?;
    let expected = case.array(&session)?.apply(&projection)?;
    assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
    Ok(())
}

/// Each filter, read from the file and evaluated on the values in memory.
#[rstest]
#[case::on_the_id(Case::Nested, gt_eq(col("id"), lit(12_345i64)))]
#[case::list_contains(Case::Primitive, list_contains(col("l"), lit(30_000i64)))]
#[case::nullable_list_contains(Case::Nullable, list_contains(col("l"), lit(7i32)))]
fn golden_file_reads_filters(#[case] case: Case, #[case] filter: Expression) -> VortexResult<()> {
    let session = new_session();
    let file = golden(&session, case)?;
    let actual = scan(&file, |scan| Ok(scan.with_filter(bind(&file, &filter)?)))?;
    let expected = filtered(&session, &case.array(&session)?, &filter)?;
    assert!(!expected.is_empty(), "the filter keeps no rows");
    assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
    Ok(())
}

/// `x -> x.<field>`, for `list_transform`.
fn element(field: &str) -> Lambda {
    Lambda::new(["x"], get_item(field, var("x"))).vortex_expect("lambda binds one parameter")
}

/// Writes the golden file of every case that has none. Run it only to add a case, never to
/// replace a file; see `goldenfiles/list_layout/README.md`.
#[test]
#[ignore = "writes golden files"]
fn write_missing_golden_files() -> VortexResult<()> {
    let session = new_session();
    for case in CASES {
        let path = case.path();
        if path.exists() {
            continue;
        }
        fs::create_dir_all(path.parent().vortex_expect("golden files have a directory"))?;
        fs::write(&path, write(case.array(&session)?)?.as_slice())?;
    }
    Ok(())
}
