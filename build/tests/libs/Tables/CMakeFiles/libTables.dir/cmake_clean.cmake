file(REMOVE_RECURSE
  "liblibTables.a"
  "liblibTables.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/libTables.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
