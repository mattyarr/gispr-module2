# [R] HOMEWORK A1 — Run this cell and screenshot the output

pkgs <- c("sf", "terra", "dplyr", "tidyr", "ggplot2",
           "tmap", "arcgisbinding", "arcgislayers")

cat(R.version.string, "\n")
cat(strrep("-", 50), "\n")
for (p in pkgs) {
  if (requireNamespace(p, quietly = TRUE)) {
    cat(sprintf("  ✓  %-20s %s\n", p, packageVersion(p)))
  } else {
    cat(sprintf("  ✗  %-20s not installed\n", p))
  }
}