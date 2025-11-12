# Boas Test Configuration
# Enable MLIR test tools
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(BOAS_TEST_DEPENDS
  FileCheck count not
  boas-opt
)

add_lit_testsuite(check-boas "Running the Boas regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${BOAS_TEST_DEPENDS}
)
set_target_properties(check-boas PROPERTIES FOLDER "Tests")

add_lit_testsuites(BOAS ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${BOAS_TEST_DEPENDS})
