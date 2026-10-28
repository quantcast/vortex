// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.google.common.base.Splitter;
import dev.vortex.api.Session;
import dev.vortex.jni.NativeFiles;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Lists the data files of a Vortex table directory.
 *
 * <p>The native listing returns every object under a prefix. Like Spark's own file index (see
 * {@code HadoopFSUtils.shouldFilterOutPathName}), this skips files and directories whose name starts with {@code _} or
 * {@code .}, such as {@code _SUCCESS} markers, {@code .crc} checksums, {@code .DS_Store} and {@code _temporary} task
 * output, unless the name contains {@code =} (a Hive-style partition directory). Files still being copied
 * ({@code ._COPYING_}) are skipped too. Only path components below the listed root are inspected, so a table may itself
 * live under a hidden directory.
 */
public final class VortexFileListing {
    private static final Pattern URI_SCHEME = Pattern.compile("^[A-Za-z][A-Za-z0-9+.-]+:");
    private static final Splitter PATH_SPLITTER = Splitter.on('/').omitEmptyStrings();

    private VortexFileListing() {}

    /** Lists the data files under {@code root}, as URIs, skipping Spark hidden files and directories. */
    public static List<String> listDataFiles(Session session, String root, Map<String, String> options) {
        int rootDepth = pathSegments(root).size();
        return NativeFiles.listFiles(session, root, options).stream()
                .filter(uri -> {
                    List<String> segments = pathSegments(uri);
                    return segments.subList(Math.min(rootDepth, segments.size()), segments.size()).stream()
                            .noneMatch(VortexFileListing::isHiddenName);
                })
                .collect(Collectors.toList());
    }

    /** Spark's own rule, from {@code HadoopFSUtils.shouldFilterOutPathName}. */
    static boolean isHiddenName(String name) {
        boolean hidden =
                (name.startsWith("_") && !name.contains("=")) || name.startsWith(".") || name.endsWith("._COPYING_");
        return hidden && !name.startsWith("_metadata") && !name.startsWith("_common_metadata");
    }

    /**
     * Splits the path of a URI, or of a bare filesystem path, into its components. A bare path is made absolute and
     * {@code .}/{@code ..} are resolved, matching how the native listing normalizes its root, so that the depth of a
     * root and of the URIs listed under it line up.
     */
    static List<String> pathSegments(String uriOrPath) {
        String path;
        Matcher scheme = URI_SCHEME.matcher(uriOrPath);
        if (scheme.find()) {
            path = uriOrPath.substring(scheme.end());
            if (path.startsWith("//")) {
                int pathStart = path.indexOf('/', 2);
                path = pathStart < 0 ? "" : path.substring(pathStart);
            }
        } else {
            path = Paths.get(uriOrPath).toAbsolutePath().toString();
        }
        List<String> segments = new ArrayList<>();
        for (String segment : PATH_SPLITTER.split(path)) {
            if (segment.equals("..")) {
                if (!segments.isEmpty()) {
                    segments.remove(segments.size() - 1);
                }
            } else if (!segment.equals(".")) {
                segments.add(segment);
            }
        }
        return segments;
    }
}
