CXX ?= $(shell command -v g++ || shell command -v clang++)
AR ?= ar
PY := $(shell command -v python3 || shell command -v python)

ifeq ($(CXX),)
  $(error "C++ compiler '$(CXX)' not found. Please install g++ or clang.")
endif
ifeq ($(AR),)
  $(error "Archiver '$(AR)' not found. Please install build-essential or binutils.")
endif
ifeq ($(PY),)
  $(error "Python not found. Please install Python or create a virtualenv first (e.g., 'python3 -m venv venv && source venv/bin/activate').")
endif

SOLVER ?= bitwuzla
INSTALL_PREFIX ?= /usr/local
TENSOR_REF ?= 1
BOOST_CPPFLAGS ?=

CXXFLAGS = -std=c++20 -Iinclude -Wall -Wextra -O2
DEPFLAGS = -MMD -MP
LDFLAGS =
ARFLAGS = rcs

ifeq ($(TENSOR_REF),1)
  CXXFLAGS += -DREFRACTIR_TENSOR_REF $(BOOST_CPPFLAGS)
else ifeq ($(TENSOR_REF),0)
  CXXFLAGS += -DREFRACTIR_TENSOR_REF_DISABLED
else
  $(error "TENSOR_REF must be 0 or 1")
endif

# Coverage support
ifeq ($(COVERAGE), 1)
  CXXFLAGS += --coverage
  LDFLAGS += --coverage
endif

# Solver support
ifeq ($(SOLVER), bitwuzla)
  ifneq ($(shell pkg-config --exists bitwuzla && echo found),found)
    $(error "Bitwuzla not found. Please install it and ensure it is discoverable via pkg-config.")
  endif
  CXXFLAGS += -DUSE_BITWUZLA $(shell pkg-config --cflags bitwuzla)
  LDFLAGS += $(shell pkg-config --libs bitwuzla)
  SOLVER_SRCS += src/solver/bitwuzla_impl.cpp
  SOLVER_IMPL_OBJ = src/solver/bitwuzla_impl.o
else ifeq ($(SOLVER), alivesmt)
  ifneq ($(shell pkg-config --exists z3 && echo found),found)
    $(error "Z3 not found. Please install it and ensure it is discoverable via pkg-config.")
  endif
  CXXFLAGS += -DUSE_ALIVESMT -Ialivesmt/include $(shell pkg-config --cflags z3)
  LDFLAGS += $(shell pkg-config --libs z3)
  ALIVESMT_SRCS = alivesmt/lib/ctx.cpp alivesmt/lib/expr.cpp alivesmt/lib/exprs.cpp alivesmt/lib/smt.cpp alivesmt/lib/solver.cpp alivesmt/lib/util.cpp
  SOLVER_SRCS += src/solver/alive_impl.cpp $(ALIVESMT_SRCS)
  SOLVER_IMPL_OBJ = src/solver/alive_impl.o $(ALIVESMT_SRCS:.cpp=.o)
else
  $(error "Unknown SOLVER: $(SOLVER). Supported: bitwuzla, alivesmt")
endif

COMMON_SRCS = src/frontend/lexer.cpp src/frontend/parser.cpp src/frontend/ast_dumper.cpp \
              src/frontend/sir_printer.cpp src/frontend/pipeline.cpp src/ast/clone.cpp \
              src/analysis/cfgbuilder.cpp src/analysis/definite_init.cpp \
              src/analysis/dominators.cpp src/analysis/reducibility.cpp \
              src/analysis/loop_info.cpp src/analysis/structurizer.cpp \
              src/analysis/structured_lowering.cpp \
              src/frontend/typechecker.cpp src/frontend/semchecker.cpp \
              src/analysis/pass_manager.cpp src/analysis/reachability.cpp \
              src/analysis/unused_name.cpp src/analysis/type_utils.cpp \
              src/analysis/state_set.cpp src/analysis/tensor.cpp \
              src/frontend/diagnostics.cpp \
              src/solver/solved_header.cpp

TEST_SRCS =
BACKEND_SRCS = src/backend/emit.cpp \
               src/backend/c_backend.cpp src/backend/wasm_backend.cpp \
               src/backend/c_expr.cpp src/backend/c_vec.cpp \
               src/backend/c_lvalue.cpp src/backend/c_types.cpp \
               src/backend/c_structured.cpp \
               src/backend/wasm_expr.cpp src/backend/wasm_vec.cpp \
               src/backend/wasm_structured.cpp \
               src/backend/wasm_lvalue.cpp src/backend/wasm_types.cpp \
               src/backend/wasm_vec_lowering_array.cpp \
               src/backend/wasm_vec_lowering_scalars.cpp \
               src/backend/wasm_vec_lowering_vecext.cpp \
               src/backend/py_backend.cpp src/backend/py_expr.cpp \
               src/backend/py_types.cpp src/backend/py_lvalue.cpp \
               src/backend/triton_backend.cpp \
               src/backend/py_vec_lowering_array.cpp \
               src/backend/py_vec_lowering_scalars.cpp \
               src/backend/py_vec_lowering_struct.cpp \
               src/backend/vec_shapes.cpp \
               src/backend/c_intrinsics.cpp src/backend/wasm_intrinsics.cpp \
               src/backend/py_intrinsics.cpp \
               src/backend/c_vec_lowering_vecext.cpp \
               src/backend/c_vec_lowering_array.cpp \
               src/backend/c_vec_lowering_scalars.cpp \
               src/backend/c_vec_lowering_struct.cpp
INTERP_IMPL_SRCS = src/interp/interpreter.cpp src/interp/intrinsics.cpp \
                   src/interp/type_layout.cpp src/interp/types.cpp \
                   src/interp/memory.cpp src/interp/values.cpp \
                   src/interp/expr.cpp src/interp/lvalue.cpp \
                   src/interp/tensor.cpp src/interp/tensor_io.cpp \
                   src/interp/tensor_numeric.cpp
INTERP_SRCS = src/symiri.cpp $(INTERP_IMPL_SRCS)
COMPILER_SRCS = src/symirc.cpp $(BACKEND_SRCS)
# Core symbolic-executor sources (distinct from the SMT backend in
# SOLVER_SRCS / SOLVER_IMPL_OBJ). Shared by symirsolve, rysmith, rylink, and
# libsymir, so every consumer picks up new TUs from one place.
SOLVER_CORE_SRCS = src/solver/solver.cpp src/solver/intrinsics.cpp \
                   src/solver/values.cpp src/solver/expr.cpp \
                   src/solver/vec.cpp src/solver/tensor.cpp src/solver/lvalue.cpp \
                   src/solver/calls.cpp src/solver/types.cpp
SOLVER_MAIN_SRCS = src/symirsolve.cpp $(SOLVER_CORE_SRCS)
SOLVER_ALL_SRCS = $(SOLVER_MAIN_SRCS) $(SOLVER_SRCS)
REIFY_SRCS = src/reify/cfg_gen.cpp src/reify/path_sampler.cpp \
             src/reify/type_gen.cpp src/reify/var_catalogue.cpp \
             src/reify/expr_gen.cpp src/reify/func_gen.cpp \
             src/reify/func_desc.cpp \
             src/reify/checksum.cpp \
             src/reify/common.cpp \
             src/reify/func_pool.cpp src/reify/cg_gen.cpp \
             src/reify/call_realize.cpp src/reify/dataflow_policy.cpp src/reify/name_alloc.cpp src/reify/state_profile.cpp \
             src/reify/transform.cpp src/reify/twin_transform.cpp \
             src/reify/twin_mini.cpp \
             src/reify/twin_probe.cpp src/reify/twin_trace.cpp \
             src/reify/twin_box.cpp src/reify/antiopt.cpp \
             src/reify/tensor_case.cpp \
             src/reify/antiopt/common.cpp src/reify/antiopt/rules_peephole.cpp \
             src/reify/antiopt/rules_mba.cpp src/reify/antiopt/rules_structure.cpp \
             src/reify/antiopt/rules_licensed.cpp \
             src/reify/antiopt/rules_control.cpp \
             src/reify/antiopt/selftest.cpp src/reify/antiopt_transform.cpp
# The tensor profile is dispatched only by rysmith and constructs the selected
# SMT backend, so keep its adapters outside the source set shared with rylink.
RYSMITH_SRCS = src/rysmith.cpp src/reify/tensor_gen.cpp \
               src/reify/tensor_graph_gen.cpp src/reify/tensor_input_policy.cpp \
               src/reify/tensor_selection.cpp src/reify/tensor_region.cpp \
               src/reify/tensor_free_form.cpp \
               $(SOLVER_CORE_SRCS) $(REIFY_SRCS) $(BACKEND_SRCS) $(INTERP_IMPL_SRCS)
# [v0.2.2] rylink links the C / WASM backends in-process so the bundle's
# FunDecl::sourceStem survives all the way to emitSplit. Driving symirc
# as a subprocess instead would parse the bundled .sir back from text
# and reset every sourceStem to "", collapsing --split-by-source to a
# single program.c.
RYLINK_SRCS = src/rylink.cpp $(REIFY_SRCS) $(BACKEND_SRCS) $(INTERP_IMPL_SRCS)
# rytwin builds twin bodies from the profiled trace and decides guards by
# interpretation, so it links neither the symbolic executor nor an SMT
# backend — unlike rysmith and rylink.
RYTWIN_SRCS = src/rytwin.cpp $(REIFY_SRCS) $(BACKEND_SRCS) $(INTERP_IMPL_SRCS)

COMMON_OBJS = $(COMMON_SRCS:.cpp=.o)
TEST_OBJS = $(TEST_SRCS:.cpp=.o)
INTERP_OBJS = $(INTERP_SRCS:.cpp=.o)
COMPILER_OBJS = $(COMPILER_SRCS:.cpp=.o)
SOLVER_OBJS = $(SOLVER_MAIN_SRCS:.cpp=.o) $(SOLVER_IMPL_OBJ)
RYSMITH_OBJS = $(RYSMITH_SRCS:.cpp=.o) $(SOLVER_IMPL_OBJ)
RYLINK_OBJS = $(RYLINK_SRCS:.cpp=.o)
RYTWIN_OBJS = $(RYTWIN_SRCS:.cpp=.o)

TARGET_INTERP = symiri
TARGET_COMPILER = symirc
TARGET_SOLVER = symirsolve
TARGET_RYSMITH = rysmith
TARGET_RYLINK = rylink
TARGET_RYTWIN = rytwin

BUILD_DIR = build
BIN_DIR = $(BUILD_DIR)/bin
LIB_DIR = $(BUILD_DIR)/lib
INC_DIR = $(BUILD_DIR)/include

LIB_NAME = libsymir.a
# The interpreter translation units, tensor transport included: a consumer that
# interprets a tensor program resolves readTensorArguments, writeTensorResult,
# and the numerical tensor helpers from the archive, so leaving them out is a
# link error for the first external user rather than an archive-size choice.
LIBRARY_OBJS = $(COMMON_OBJS) \
               src/interp/interpreter.o \
               src/interp/intrinsics.o \
               src/interp/type_layout.o \
               src/interp/types.o \
               src/interp/memory.o \
               src/interp/values.o \
               src/interp/expr.o \
               src/interp/lvalue.o \
               src/interp/tensor.o \
               src/interp/tensor_io.o \
               src/interp/tensor_numeric.o \
               src/backend/c_backend.o \
               src/backend/c_expr.o \
               src/backend/c_vec.o \
               src/backend/c_lvalue.o \
               src/backend/c_types.o \
               src/backend/c_intrinsics.o \
               src/backend/wasm_intrinsics.o \
               src/backend/c_vec_lowering_vecext.o \
               src/backend/c_vec_lowering_array.o \
               src/backend/c_vec_lowering_scalars.o \
               src/backend/c_vec_lowering_struct.o \
               src/backend/wasm_backend.o \
               src/backend/wasm_expr.o \
               src/backend/wasm_vec.o \
               src/backend/wasm_structured.o \
               src/backend/wasm_lvalue.o \
               src/backend/wasm_types.o \
               src/backend/wasm_vec_lowering_array.o \
               src/backend/wasm_vec_lowering_scalars.o \
               src/backend/wasm_vec_lowering_vecext.o \
               src/backend/py_backend.o \
               src/backend/py_expr.o \
               src/backend/py_types.o \
               src/backend/py_lvalue.o \
               src/backend/py_vec_lowering_array.o \
               src/backend/py_vec_lowering_scalars.o \
               src/backend/py_vec_lowering_struct.o \
               src/backend/vec_shapes.o \
               src/backend/py_intrinsics.o \
               $(SOLVER_CORE_SRCS:.cpp=.o) \
               $(SOLVER_IMPL_OBJ)

DEPS = $(COMMON_OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(INTERP_OBJS:.o=.d) \
       $(COMPILER_OBJS:.o=.d) $(SOLVER_OBJS:.o=.d) $(RYSMITH_OBJS:.o=.d) \
       $(RYLINK_OBJS:.o=.d) $(RYTWIN_OBJS:.o=.d)
-include $(DEPS)

.PHONY: test-tensor-discovery-batch test-tensor-tail-shapes test-tensor-region-family test-tensor-region-options all clean test test-unit test-frontend test-analysis test-interp test-backends test-cross-validation test-solver test-reify test-tensor-coverage test-tensor-directed test-tensor-layout test-tensor-probe test-tensor-profile test-tensor-window test-tensor-v2-profile test-tensor-input-policy test-tensor-v4-generation test-tensor-v4 test-tensor-frontend test-tensor-exact test-tensor-numeric test-tensor-solver test-tensor-emit test-tensor-runner test-tensor-build test-tensor-regressions test-tensor-mutants test-tensor-historical test-tensor-evaluation test-tensor-fixed-study test-tensor-boundary-sensitivity test-tensor-campaign-summary test-tensor-reducer test-tensor-property-pilot test-tensor-graph-audit test-tensor-region test-tensor-free-form test-tensor-graph-region test-tensor-generated-corpus test-tensor-region-generation test-tensor-fpsan-model test-tensor-gluon-layouts test-tensor-gluon-lowering test-tensor-three-path test-tensor-compiler-observation test-tensor-fp-contracts test-triton-fp-controls test-triton-fpsan test-triton-gluon test-triton-gluon-study test-triton-exact-loop test-tensor test-triton test-triton-v2 test-library test-solver-outcomes cross-validation build install

all: $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_SOLVER) $(TARGET_RYSMITH) $(TARGET_RYLINK) $(TARGET_RYTWIN)

$(TARGET_INTERP): $(COMMON_OBJS) $(INTERP_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_COMPILER): $(COMMON_OBJS) $(COMPILER_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

TENSOR_EMIT_API = $(BIN_DIR)/tensor_emit_api

$(TENSOR_EMIT_API): test/unit/tensor_emit_api.cpp $(COMMON_OBJS) $(BACKEND_SRCS:.cpp=.o) src/reify/tensor_case.o
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

TENSOR_INPUT_POLICY_API = $(BIN_DIR)/tensor_input_policy_api
TENSOR_INPUT_POLICY_OBJS = $(COMMON_OBJS) $(filter-out src/rysmith.o,$(RYSMITH_OBJS))

$(TENSOR_INPUT_POLICY_API): test/unit/tensor_input_policy_api.cpp $(TENSOR_INPUT_POLICY_OBJS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test-tensor-input-policy: $(TARGET_INTERP) $(TENSOR_INPUT_POLICY_API)
	REFRACTIR_TENSOR_INPUT_POLICY_API=$(TENSOR_INPUT_POLICY_API) \
	  REFRACTIR_SYMIRI=./$(TARGET_INTERP) \
	  $(PY) -m unittest \
	    test.unit.test_tensor_generation.TensorInputPolicyTests.test_boundary_same_shape_parameters_have_independent_samples \
	    test.unit.test_tensor_generation.TensorInputPolicyTests.test_boundary_equal_element_count_different_shapes_are_independent \
	    test.unit.test_tensor_generation.TensorInputPolicyTests.test_boundary_rows_do_not_repeat_every_four_elements \
	    test.unit.test_tensor_generation.TensorInputPolicyTests.test_boundary_seed_does_not_repeat_on_a_four_seed_cycle \
	    test.unit.test_tensor_generation.TensorInputPolicyTests.test_boundary_proposals_distinguish_replaced_operands_for_i32_and_f32 \
	    test.unit.test_tensor_generation.TensorInputPolicyTests.test_boundary_singleton_is_reproducible_and_in_domain

TENSOR_WINDOW_API = $(BIN_DIR)/tensor_window_api

$(TENSOR_WINDOW_API): test/unit/tensor_window_api.cpp $(COMMON_OBJS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test-tensor-window: $(TENSOR_WINDOW_API)
	REFRACTIR_TENSOR_WINDOW_API=$(TENSOR_WINDOW_API) \
	  $(PY) -m unittest test.unit.test_tensor_window

TENSOR_REGION_API = $(BIN_DIR)/tensor_region_api

$(TENSOR_REGION_API): test/unit/tensor_region_api.cpp src/reify/tensor_region.cpp include/reify/tensor_region.hpp
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ test/unit/tensor_region_api.cpp src/reify/tensor_region.cpp

test-tensor-region: $(TENSOR_REGION_API) $(TARGET_INTERP)
	REFRACTIR_TENSOR_REGION_API=$(TENSOR_REGION_API) \
	  REFRACTIR_SYMIRI=./$(TARGET_INTERP) \
	  $(PY) -m unittest test.unit.test_tensor_region test.unit.test_tensor_region_integration

TENSOR_FREE_FORM_API = $(BIN_DIR)/tensor_free_form_api
TENSOR_FREE_FORM_SRCS = src/reify/tensor_free_form.cpp src/reify/tensor_region.cpp \
                        src/analysis/tensor.cpp src/analysis/type_utils.cpp \
                        src/analysis/cfgbuilder.cpp src/analysis/reducibility.cpp \
                        src/analysis/dominators.cpp src/analysis/loop_info.cpp \
                        src/frontend/diagnostics.cpp

TENSOR_FREE_FORM_EXPORT = $(BIN_DIR)/tensor_free_form_export

$(TENSOR_FREE_FORM_API): test/unit/tensor_free_form_api.cpp $(TENSOR_FREE_FORM_SRCS) include/reify/tensor_free_form.hpp include/reify/tensor_region.hpp
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ test/unit/tensor_free_form_api.cpp $(TENSOR_FREE_FORM_SRCS)

$(TENSOR_FREE_FORM_EXPORT): test/unit/tensor_free_form_export.cpp $(TENSOR_FREE_FORM_SRCS) include/reify/tensor_free_form.hpp include/reify/tensor_region.hpp
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ test/unit/tensor_free_form_export.cpp $(TENSOR_FREE_FORM_SRCS)

test-tensor-free-form: $(TARGET_INTERP) $(TARGET_COMPILER) $(TENSOR_FREE_FORM_API) $(TENSOR_FREE_FORM_EXPORT)
	REFRACTIR_TENSOR_FREE_FORM_API=$(TENSOR_FREE_FORM_API) \
	  REFRACTIR_TENSOR_FREE_FORM_EXPORT=$(TENSOR_FREE_FORM_EXPORT) \
	  REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  $(PY) -m unittest test.unit.test_tensor_free_form \
	    test.unit.test_tensor_free_form_inputs test.unit.test_tensor_free_form_case \
	    test.unit.test_tensor_target_case test.unit.test_tensor_target_run \
	    test.unit.test_tensor_target_reference test.unit.test_tensor_state_diagnostics \
	    test.unit.test_tensor_tranche test.unit.test_moving_window_discovery \
	    test.unit.test_moving_window_repairs test.unit.test_moving_window_closure \
	    test.unit.test_moving_window_review_fixes test.unit.test_moving_window_delivery \
	    test.unit.test_moving_window_followup test.unit.test_moving_window_sweep

TENSOR_GRAPH_REGION_API = $(BIN_DIR)/tensor_graph_region_api
TENSOR_GRAPH_REGION_EXPORT = $(BIN_DIR)/tensor_graph_region_export

TENSOR_GRAPH_REGION_SRCS = src/reify/tensor_graph_gen.cpp src/reify/tensor_region.cpp src/analysis/tensor.cpp src/analysis/type_utils.cpp src/analysis/cfgbuilder.cpp src/analysis/reducibility.cpp src/analysis/dominators.cpp src/analysis/loop_info.cpp src/frontend/diagnostics.cpp

$(TENSOR_GRAPH_REGION_API): test/unit/tensor_graph_region_api.cpp $(TENSOR_GRAPH_REGION_SRCS) include/reify/tensor_graph_gen.hpp include/reify/tensor_region.hpp include/reify/tensor_region_family.hpp
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ test/unit/tensor_graph_region_api.cpp $(TENSOR_GRAPH_REGION_SRCS)

$(TENSOR_GRAPH_REGION_EXPORT): test/unit/tensor_graph_region_export.cpp $(TENSOR_GRAPH_REGION_SRCS) include/reify/tensor_graph_gen.hpp include/reify/tensor_region.hpp include/reify/tensor_region_family.hpp
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ test/unit/tensor_graph_region_export.cpp $(TENSOR_GRAPH_REGION_SRCS)

test-tensor-graph-region: $(TENSOR_GRAPH_REGION_API)
	REFRACTIR_TENSOR_GRAPH_REGION_API=$(TENSOR_GRAPH_REGION_API) $(PY) -m unittest test.unit.test_tensor_graph_region

test-tensor-generated-corpus: $(BIN_DIR)/tensor_graph_region_export
	REFRACTIR_TENSOR_GRAPH_EXPORT=$(TENSOR_GRAPH_REGION_EXPORT) $(PY) -m unittest test.unit.test_tensor_generated_region_corpus

# Explicit long-axis region configurations: reachability, graph families, and
# the exact certificate each admitted configuration has to carry.
test-tensor-region-generation: $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_SOLVER) $(TENSOR_GRAPH_REGION_EXPORT)
	REFRACTIR_TENSOR_GRAPH_EXPORT=$(TENSOR_GRAPH_REGION_EXPORT) \
	  REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  REFRACTIR_SYMIR_SOLVE=./$(TARGET_SOLVER) \
	  $(PY) -m unittest test.unit.test_tensor_region_generation \
	    test.unit.test_tensor_region_exact_family \
	    test.unit.test_tensor_input_proposals

test-tensor-fpsan-model:
	$(PY) -m unittest test.unit.test_tensor_fpsan_model

test-tensor-gluon-layouts:
	$(PY) -m unittest test.unit.test_tensor_gluon_layouts

test-tensor-gluon-lowering: $(TENSOR_GRAPH_REGION_EXPORT)
	REFRACTIR_TENSOR_GRAPH_EXPORT=$(TENSOR_GRAPH_REGION_EXPORT) \
	  $(PY) -m unittest test.unit.test_tensor_gluon_lowering test.unit.test_tensor_gluon_plan \
	    test.unit.test_tensor_gluon_generated_region \
	    test.unit.test_tensor_generated_region_runner \
	    test.unit.test_tensor_emission

test-tensor-three-path: $(TARGET_INTERP) $(TENSOR_GRAPH_REGION_EXPORT)
	REFRACTIR_TENSOR_GRAPH_EXPORT=$(TENSOR_GRAPH_REGION_EXPORT) \
	  $(PY) -m unittest test.unit.test_tensor_emission test.unit.test_tensor_reference \
	    test.unit.test_tensor_three_way test.unit.test_tensor_tranche \
	    test.unit.test_tensor_input_proposals_i32 test.unit.test_tensor_fp_diagnostics \
	    test.unit.test_tensor_target_evidence

test-tensor-compiler-observation:
	$(PY) -m unittest test.unit.test_tensor_compiler_observation \
	  test.unit.test_tensor_evidence_records \
	  test.unit.test_tensor_rematerialization_evidence

test-tensor-fp-contracts:
	REFRACTIR_TEST_CXX="$(CXX)" $(PY) -m unittest test.unit.test_ast_type_construction test.unit.test_tensor_fp32_reference test.unit.test_tensor_fp32_edge_contract test.unit.test_tensor_ordinary_fp_controls test.unit.test_tensor_route_protocol

test-triton-fp-controls:
	mkdir -p $(BUILD_DIR)/fpsan/ordinary-ir
	$(PY) -m test.tensor.fpsan.run_ordinary_fp_controls \
	  --output $(BUILD_DIR)/fpsan/ordinary-controls.json \
	  --ir-dir $(BUILD_DIR)/fpsan/ordinary-ir

test-triton-fpsan: test-triton-fp-controls
	mkdir -p $(BUILD_DIR)/fpsan/instrumented-ir
	$(PY) -m test.tensor.fpsan.run_experiment \
	  --output $(BUILD_DIR)/fpsan/experiment.json \
	  --ir-dir $(BUILD_DIR)/fpsan/instrumented-ir

test-triton-gluon:
	mkdir -p $(BUILD_DIR)/gluon/ir
	$(PY) -m test.tensor.gluon.run_exact_loop \
	  --output $(BUILD_DIR)/gluon/exact-loop.json \
	  --ir-dir $(BUILD_DIR)/gluon/ir

TENSOR_GLUON_STUDY_DIR ?= $(BUILD_DIR)/gluon/fixed-domain-study-$(shell date +%Y%m%d-%H%M%S)

test-triton-gluon-study: $(TARGET_SOLVER)
	$(PY) -m test.tensor.gluon.run_selection_study \
	  --output-dir $(TENSOR_GLUON_STUDY_DIR) \
	  --symirsolve ./$(TARGET_SOLVER)

TENSOR_EXACT_CASE_DIR ?= build/tensor_cases/exact-composed-loop

test-triton-exact-loop: $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_SOLVER)
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  REFRACTIR_SYMIR_SOLVE=./$(TARGET_SOLVER) \
	  REFRACTIR_REVISION='$(TENSOR_SOURCE_REVISION)' \
	  REFRACTIR_WORKTREE_DIRTY='$(TENSOR_WORKTREE_DIRTY)' \
	  REFRACTIR_COMPILER_FLAGS='$(TENSOR_COMPILER_FLAGS)' \
	  $(PY) -m test.tensor.exact.run_integration \
	    --output-dir $(TENSOR_EXACT_CASE_DIR) \
	    --symiri ./$(TARGET_INTERP) --symirc ./$(TARGET_COMPILER) \
	    --symirsolve ./$(TARGET_SOLVER)

# A link-and-run consumer of the installed archive. Building libsymir.a only
# proves that its objects were archived, not that they resolve each other.
LIBRARY_CONSUMER = $(BIN_DIR)/symir_library_consumer

$(LIBRARY_CONSUMER): test/unit/symir_library_consumer.cpp $(LIB_DIR)/$(LIB_NAME)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ test/unit/symir_library_consumer.cpp \
	  -L$(LIB_DIR) -lsymir $(LDFLAGS)

test-library: $(LIBRARY_CONSUMER)
	./$(LIBRARY_CONSUMER)

# A controlled backend for the solver-outcome classification tests: timing a
# real solver is not a reliable way to produce a timeout.
SOLVER_OUTCOME_PROBE = $(BIN_DIR)/solver_outcome_probe

# The probe reaches the tensor generation path as well, so it links the same
# objects the generator does, minus the tool's own entry point.
PROBE_OBJS = $(filter-out src/rysmith.o,$(RYSMITH_OBJS))

$(SOLVER_OUTCOME_PROBE): test/unit/solver_outcome_probe.cpp $(COMMON_OBJS) $(PROBE_OBJS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ test/unit/solver_outcome_probe.cpp $(COMMON_OBJS) $(PROBE_OBJS) $(LDFLAGS)

test-solver-outcomes: $(SOLVER_OUTCOME_PROBE) $(TARGET_SOLVER)
	REFRACTIR_SOLVER_OUTCOME_PROBE=./$(SOLVER_OUTCOME_PROBE) \
	  REFRACTIR_SYMIRSOLVE=./$(TARGET_SOLVER) \
	  $(PY) -m unittest test.unit.test_tensor_selection_outcomes \
	                     test.unit.test_tensor_selection_deadline

$(TARGET_SOLVER): $(COMMON_OBJS) $(SOLVER_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_RYSMITH): $(COMMON_OBJS) $(RYSMITH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# [v0.2.2] rylink also depends on the solver objects: it doesn't call
# the SMT solver itself, but the reify libs (and SIRPrinter) include
# solver headers and reference its types, so the linker needs the
# symbols. We don't link the Bitwuzla impl since rylink never solves.
$(TARGET_RYLINK): $(COMMON_OBJS) $(RYLINK_OBJS) $(SOLVER_CORE_SRCS:.cpp=.o)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_RYTWIN): $(COMMON_OBJS) $(RYTWIN_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c -o $@ $<

build: all $(LIB_DIR)/$(LIB_NAME)
	mkdir -p $(BIN_DIR) $(INC_DIR)
	cp $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_SOLVER) $(TARGET_RYSMITH) $(TARGET_RYLINK) $(BIN_DIR)/
	cp -r include/* $(INC_DIR)/

install: build
	mkdir -p $(INSTALL_PREFIX)/bin
	mkdir -p $(INSTALL_PREFIX)/lib
	mkdir -p $(INSTALL_PREFIX)/include/refractir
	cp -fp $(BIN_DIR)/* $(INSTALL_PREFIX)/bin/
	cp -fp $(LIB_DIR)/* $(INSTALL_PREFIX)/lib/
	cp -rfp $(INC_DIR)/* $(INSTALL_PREFIX)/include/refractir/

$(LIB_DIR)/$(LIB_NAME): $(LIBRARY_OBJS)
	mkdir -p $(LIB_DIR)
	$(AR) $(ARFLAGS) $@ $^

clean:
	rm -f $(COMMON_OBJS) $(TEST_OBJS) $(INTERP_OBJS) $(COMPILER_OBJS) $(SOLVER_OBJS) $(RYSMITH_OBJS) $(RYLINK_OBJS) $(RYTWIN_OBJS) $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_SOLVER) $(TARGET_RYSMITH) $(TARGET_RYLINK) $(TARGET_RYTWIN)
	rm -rf $(BUILD_DIR)
	find src alivesmt -name "*.d" -delete
	find . -name "*.gcno" -delete
	find . -name "*.gcda" -delete
	find . -name "*.gcov" -delete

# [v0.2.2] Unit tests — end-to-end Python scripts that exercise the
# CLI surface of individual binaries (positional args, descriptors,
# split-by-source output, etc.). They don't go through the `.sir`
# test runner in test/lib because they need to assert on the binary's
# stdout / sidecar files / output directory layout.
test-unit: $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_SOLVER) $(TARGET_RYSMITH) $(TARGET_RYLINK) $(TARGET_RYTWIN)
	$(PY) -m unittest test.unit.test_unused_name_clang_portability
	$(PY) -m test.unit.run_param_features_tests ./$(TARGET_INTERP) ./$(TARGET_COMPILER) ./$(TARGET_SOLVER)
	$(PY) -m test.unit.run_structured_c_tests ./$(TARGET_COMPILER)
	$(PY) -m test.unit.run_structured_wasm_tests ./$(TARGET_COMPILER)
	$(PY) -m test.unit.run_rysmith_tests ./$(TARGET_RYSMITH) ./$(TARGET_INTERP) ./$(TARGET_COMPILER)
	$(PY) -m test.unit.run_rylink_tests ./$(TARGET_RYLINK) ./$(TARGET_RYSMITH) ./$(TARGET_INTERP)
	$(PY) -m test.unit.run_rytwin_tests ./$(TARGET_RYTWIN) ./$(TARGET_RYSMITH) ./$(TARGET_INTERP)

# Tensor environment/solver supervisor tests are host-only and use fake cvc5
# executables; real GPU and cvc5 feasibility probes are invoked separately.
test-tensor-coverage:
	$(PY) -m unittest test.unit.test_tensor_selection.TensorCoverageTests

test-tensor-directed:
	$(PY) -m unittest test.unit.test_tensor_selection.TensorDirectedTests

test-tensor-layout:
	$(PY) -m unittest test.unit.test_tensor_layout

# Rank-two logical tails in the compositional generator: reachability, padded
# tiles, complete logical replay, and the masked kernel that reaches the device.
test-tensor-tail-shapes: $(TARGET_RYSMITH) $(TARGET_INTERP) $(TARGET_COMPILER)
	REFRACTIR_RYSMITH=./$(TARGET_RYSMITH) REFRACTIR_SYMIRI=./$(TARGET_INTERP) \
	  REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  $(PY) -m unittest test.unit.test_tensor_tail_shapes

# Normal-generator access to the bounded loop-region families: the CLI option,
# its refusals, and the effect on the emitted source.
test-tensor-region-family: $(TARGET_RYSMITH) $(TARGET_INTERP) $(TENSOR_GRAPH_REGION_EXPORT)
	REFRACTIR_RYSMITH=./$(TARGET_RYSMITH) REFRACTIR_SYMIRI=./$(TARGET_INTERP) \
	  REFRACTIR_TENSOR_GRAPH_EXPORT=$(TENSOR_GRAPH_REGION_EXPORT) \
	  $(PY) -m unittest test.unit.test_tensor_region_family_option

# The region's shape, trip count, and root arithmetic on the normal generation
# path: the values the case records, and the domain a request is refused by.
test-tensor-region-options: $(TARGET_RYSMITH)
	REFRACTIR_RYSMITH=./$(TARGET_RYSMITH) \
	  $(PY) -m unittest test.unit.test_tensor_region_options

test-tensor-probe:
	$(PY) -m unittest test.unit.test_tensor_probe

test-tensor-profile: $(TARGET_INTERP) $(TARGET_COMPILER)
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  $(PY) -m unittest test.unit.test_tensor_profile

test-tensor-v2-profile: $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_RYSMITH) $(TENSOR_INPUT_POLICY_API)
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  REFRACTIR_RYSMITH=./$(TARGET_RYSMITH) REFRACTIR_TENSOR_INPUT_POLICY_API=$(TENSOR_INPUT_POLICY_API) \
	  $(PY) -m unittest test.unit.test_tensor_profile test.unit.test_tensor_generation \
	    test.unit.test_tensor_case.TensorV3CaseTests test.unit.test_tensor_runner \
	    test.unit.test_tensor_selection

test-tensor-v4-generation: $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_RYSMITH)
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  REFRACTIR_RYSMITH=./$(TARGET_RYSMITH) \
	  $(PY) -m unittest test.unit.test_tensor_generation \
	    test.unit.test_tensor_case.TensorCaseV4Tests

test-tensor-v4: $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_RYSMITH) $(TENSOR_EMIT_API)
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  REFRACTIR_RYSMITH=./$(TARGET_RYSMITH) REFRACTIR_EMIT_API=$(TENSOR_EMIT_API) \
	  $(PY) -m unittest test.unit.test_tensor_frontend test.unit.test_tensor_profile \
	    test.unit.test_tensor_semantics test.unit.test_tensor_emit \
	    test.unit.test_tensor_case.TensorCaseV4Tests \
	    test.unit.test_tensor_case.TensorV4CalibrationTests test.unit.test_tensor_generation \
	    test.unit.test_tensor_runner

test-tensor-frontend: $(TARGET_INTERP) $(TARGET_COMPILER) $(TENSOR_EMIT_API)
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  REFRACTIR_EMIT_API=$(TENSOR_EMIT_API) $(PY) -m unittest test.unit.test_tensor_frontend

test-tensor-exact: $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_SOLVER) $(TENSOR_GRAPH_REGION_EXPORT)
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  REFRACTIR_SYMIR_SOLVE=./$(TARGET_SOLVER) $(PY) -m unittest \
	  test.unit.test_tensor_exact_cli test.unit.test_tensor_exact_composed \
	  test.unit.test_tensor_exact_builder \
	  test.unit.test_tensor_exact_selection_evidence \
	  test.unit.test_tensor_exact_region_case \
	  test.unit.test_tensor_companion_derivation \
	  test.unit.test_tensor_unrolled_companion \
	  test.unit.test_tensor_semantics.TensorExactTests

test-tensor-numeric: $(TARGET_INTERP)
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) $(PY) -m unittest test.unit.test_tensor_semantics.TensorNumericTests

test-tensor-solver: $(TARGET_SOLVER)
	REFRACTIR_SYMIR_SOLVE=./$(TARGET_SOLVER) $(PY) -m unittest test.unit.test_tensor_solver

test-tensor-emit: $(TARGET_COMPILER) $(TENSOR_EMIT_API)
	REFRACTIR_SYMIRC=./$(TARGET_COMPILER) REFRACTIR_EMIT_API=$(TENSOR_EMIT_API) \
	  $(PY) -m unittest test.unit.test_tensor_emit

test-tensor-runner: $(TARGET_INTERP) $(TARGET_COMPILER)
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  $(PY) -m unittest test.unit.test_tensor_runner \
	    test.unit.test_tensor_execution test.unit.test_triton_worker_timing

test-tensor-build:
	REFRACTIR_TEST_SOLVER='$(SOLVER)' $(PY) -m unittest test.unit.test_tensor_build

# Bounded dependency audit of the compositional builder: 256 seeds per profile
# at the default limits, measuring topology and operator identity separately.
# The large report is written outside the repository by the module's CLI.
test-tensor-graph-audit:
	REFRACTIR_RYSMITH=./$(TARGET_RYSMITH) REFRACTIR_SYMIRI=./$(TARGET_INTERP) \
	  $(PY) -m unittest test.unit.test_tensor_graph_audit

# Host-only tensor suite. GPU execution is deliberately opt-in via test-triton.
test-tensor: $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_SOLVER) $(TARGET_RYSMITH) $(TENSOR_EMIT_API)
	$(MAKE) test-tensor-region test-tensor-free-form test-tensor-graph-region test-tensor-generated-corpus test-tensor-region-generation test-tensor-fpsan-model test-tensor-gluon-layouts test-tensor-gluon-lowering test-tensor-compiler-observation test-tensor-fp-contracts test-tensor-fixed-study test-tensor-boundary-sensitivity test-tensor-campaign-summary test-tensor-discovery-batch test-tensor-layout test-tensor-tail-shapes test-tensor-region-family test-tensor-region-options test-tensor-probe test-tensor-frontend test-tensor-exact test-tensor-numeric \
	  test-tensor-solver test-tensor-emit test-tensor-window test-tensor-regressions \
	  test-tensor-mutants test-tensor-historical test-tensor-evaluation \
	  test-tensor-reducer test-tensor-property-pilot
	$(MAKE) test-tensor-v2-profile
	$(MAKE) test-tensor-v4
	$(MAKE) test-tensor-build

test-tensor-historical:
	$(PY) -m unittest test.unit.test_tensor_regressions.HistoricalReplayTests

test-tensor-evaluation:
	$(PY) -m unittest test.unit.test_tensor_evaluation

test-tensor-fixed-study:
	$(PY) -m unittest test.unit.test_tensor_fixed_study test.unit.test_tensor_fixed_selection test.unit.test_tensor_generated_selection_study

test-tensor-boundary-sensitivity: $(TARGET_INTERP) $(TENSOR_INPUT_POLICY_API)
	REFRACTIR_TENSOR_INPUT_POLICY_API=$(TENSOR_INPUT_POLICY_API) \
	  REFRACTIR_SYMIRI=./$(TARGET_INTERP) \
	  $(PY) -m unittest test.unit.test_tensor_boundary_sensitivity \
	    test.unit.test_tensor_boundary_sensitivity_driver

test-tensor-campaign-summary:
	$(PY) -m unittest test.unit.test_continuous_tensor_campaign

# The bounded discovery batch driver's host-side instrument: the frozen
# allocation, the freeze refusal, the seed freshness, the observation reading,
# and the compaction. The batch itself needs the device and runs separately.
test-tensor-discovery-batch:
	$(PY) -m unittest test.unit.test_track_a_discovery_batch

test-tensor-reducer:
	$(PY) -m unittest test.unit.test_tensor_reducer

# Host-only tests for frozen graph generation, input selection, and case export.
test-tensor-property-pilot: $(TARGET_INTERP) $(TARGET_SOLVER)
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIR_SOLVE=./$(TARGET_SOLVER) \
	  $(PY) -m unittest test.unit.test_tensor_property_pilot

# The loop is serial: this pilot uses one V100 and never launches cases concurrently.
# These can be supplied by a remote/container caller whose source checkout does
# not carry .git metadata. Defaults still describe a local repository build.
TENSOR_SOURCE_REVISION ?= $(shell git rev-parse HEAD 2>/dev/null || echo unavailable)
TENSOR_WORKTREE_DIRTY ?= $(shell if test -n "$$(git status --porcelain 2>/dev/null)"; then echo true; else echo false; fi)
TENSOR_COMPILER_FLAGS ?= $(CXX) $(CXXFLAGS)
TENSOR_REGRESSION_DIR ?= build/tensor_cases/regressions

test-tensor-regressions:
	$(PY) -m unittest test.unit.test_tensor_regressions.TensorRegressionMatrixTests

test-tensor-mutants:
	$(PY) -m unittest test.unit.test_triton_mutants

TENSOR_V2_CASE_DIR ?= build/tensor_cases/v2-campaign
TENSOR_V2_CASE_COUNT ?= 10
TENSOR_V2_SEED ?= 0

test-triton-v2: $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_RYSMITH)
	./$(TARGET_RYSMITH) --profile tensor --target sir --tensor-profile-version v2 \
	  --tensor-dtype auto --seed $(TENSOR_V2_SEED) -n $(TENSOR_V2_CASE_COUNT) \
	  --output-dir $(TENSOR_V2_CASE_DIR)
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  REFRACTIR_REVISION='$(TENSOR_SOURCE_REVISION)' \
	  REFRACTIR_WORKTREE_DIRTY='$(TENSOR_WORKTREE_DIRTY)' \
	  REFRACTIR_COMPILER_FLAGS='$(TENSOR_COMPILER_FLAGS)' \
	  $(PY) -m test.lib.run_triton_tests --suite $(TENSOR_V2_CASE_DIR) --mode interpret
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  REFRACTIR_REVISION='$(TENSOR_SOURCE_REVISION)' \
	  REFRACTIR_WORKTREE_DIRTY='$(TENSOR_WORKTREE_DIRTY)' \
	  REFRACTIR_COMPILER_FLAGS='$(TENSOR_COMPILER_FLAGS)' \
	  $(PY) -m test.lib.run_triton_tests --suite $(TENSOR_V2_CASE_DIR) \
	    --mode compiled --probe-warps

test-triton: $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_RYSMITH)
	$(PY) -m test.lib.build_tensor_smoke_cases --output-dir build/tensor_cases/hand
	$(PY) -m test.lib.build_tensor_regression_cases --output-dir $(TENSOR_REGRESSION_DIR)
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  REFRACTIR_REVISION='$(TENSOR_SOURCE_REVISION)' \
	  REFRACTIR_WORKTREE_DIRTY='$(TENSOR_WORKTREE_DIRTY)' \
	  REFRACTIR_COMPILER_FLAGS='$(TENSOR_COMPILER_FLAGS)' \
	  $(PY) -m test.lib.run_triton_tests --suite build/tensor_cases/hand --mode interpret
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  REFRACTIR_REVISION='$(TENSOR_SOURCE_REVISION)' \
	  REFRACTIR_WORKTREE_DIRTY='$(TENSOR_WORKTREE_DIRTY)' \
	  REFRACTIR_COMPILER_FLAGS='$(TENSOR_COMPILER_FLAGS)' \
	  $(PY) -m test.lib.run_triton_tests --suite $(TENSOR_REGRESSION_DIR) --mode interpret
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  REFRACTIR_REVISION='$(TENSOR_SOURCE_REVISION)' \
	  REFRACTIR_WORKTREE_DIRTY='$(TENSOR_WORKTREE_DIRTY)' \
	  REFRACTIR_COMPILER_FLAGS='$(TENSOR_COMPILER_FLAGS)' \
	  $(PY) -m test.lib.run_triton_tests --suite build/tensor_cases/hand --mode compiled
	REFRACTIR_SYMIRI=./$(TARGET_INTERP) REFRACTIR_SYMIRC=./$(TARGET_COMPILER) \
	  REFRACTIR_REVISION='$(TENSOR_SOURCE_REVISION)' \
	  REFRACTIR_WORKTREE_DIRTY='$(TENSOR_WORKTREE_DIRTY)' \
	  REFRACTIR_COMPILER_FLAGS='$(TENSOR_COMPILER_FLAGS)' \
	  $(PY) -m test.lib.run_triton_tests --suite $(TENSOR_REGRESSION_DIR) --mode compiled
	$(PY) -m test.lib.triton_mutants --suite $(TENSOR_REGRESSION_DIR) \
	  --output-dir build/tensor_cases/mutants --mode compiled
	$(MAKE) test-triton-v2

# Integration tests, grouped by the component under test. Each component
# target is callable on its own (e.g. `make test-frontend`) so a developer
# touching one area can run just the relevant suite. `make test` invokes
# them in dependency order via `$(MAKE)` so a failure in an earlier group
# halts the run with the right exit code and the per-group banners stay
# legible in CI logs.

# Frontend: lexer, parser, CFG builder, type checker, sem checker, and
# reducibility analyses (dominator trees, loop forests, control trees).
test-frontend: $(TARGET_INTERP) $(TARGET_COMPILER)
	$(PY) -m test.lib.run_interp_tests test/lexer ./$(TARGET_INTERP) --check
	$(PY) -m test.lib.run_interp_tests test/parser ./$(TARGET_INTERP) --check
	$(PY) -m test.lib.run_interp_tests test/cfgbuilder ./$(TARGET_INTERP) --check
	$(PY) -m test.lib.run_interp_tests test/typechecker ./$(TARGET_INTERP) --check
	$(PY) -m test.lib.run_interp_tests test/semchecker ./$(TARGET_INTERP) --check
	$(PY) -m test.lib.run_reducibility_tests test/reducibility ./$(TARGET_COMPILER)

# Interpreter: end-to-end .sir execution.
test-interp: $(TARGET_INTERP)
	$(PY) -m test.lib.run_interp_tests test/interp ./$(TARGET_INTERP)

# Backends: C / WASM emission + the C preamble. Cross-validation lives in
# its own target below so it can be run on its own.
test-backends: $(TARGET_INTERP) $(TARGET_COMPILER)
	$(PY) -m test.lib.run_c_preamble_test ./$(TARGET_COMPILER)
	$(PY) -m test.lib.run_compiler_tests test/ ./$(TARGET_COMPILER) --target c
	$(PY) -m test.lib.run_compiler_tests test/ ./$(TARGET_COMPILER) --target c --structured
	$(PY) -m test.lib.run_compiler_tests test/ ./$(TARGET_COMPILER) --target wasm
	$(PY) -m test.lib.run_compiler_tests test/ ./$(TARGET_COMPILER) --target wasm --structured
	$(PY) -m test.lib.run_compiler_tests test/ ./$(TARGET_COMPILER) --target python

# Cross-validation: run every test/xval/*.sir through symiri *and* through
# the C backend + compiled-C and diff their behaviour. This is a separate
# group because it's the only suite that exercises the interpreter and the
# C backend together end-to-end, and a developer touching either side may
# want to run it on its own (`make cross-validation`).
test-cross-validation cross-validation: $(TARGET_INTERP) $(TARGET_COMPILER)
	$(PY) -m test.lib.run_xval_tests test/xval ./$(TARGET_INTERP) ./$(TARGET_COMPILER)
	$(PY) -m test.lib.run_xval_tests test/xval ./$(TARGET_INTERP) ./$(TARGET_COMPILER) "--symirc-extra=--structured-lowering --no-ub-guards"

# Solver: symirsolve + curated examples that depend on solving.
test-solver: $(TARGET_SOLVER) $(TARGET_INTERP)
	$(PY) -m test.lib.run_solver_tests test/solver ./$(TARGET_SOLVER) ./$(TARGET_INTERP)
	$(PY) -m test.lib.run_example_tests examples ./$(TARGET_SOLVER) ./$(TARGET_INTERP)

# Reify: rysmith + rylink differential generation pipeline.
# Forward `make -jN` to the test runner's own `-j` flag so the
# per-program cross-validation phase (steps 2 and 4 of every batch)
# inherits the user's requested parallelism.  Generation and batching
# stay sequential inside the script regardless.
#
# MAKEFLAGS is empty at parse-time (GNU make only populates it when a
# recipe shell is spawned) so the extraction must run inside the
# recipe.  Inside the recipe: `make -j8` produces something like
# `-j8 --jobserver-auth=3,4`; bare `make -j` is `-j …`; plain `make`
# is empty.  Grep for `-j<digits>`, default to 1 if absent — bare
# `make -j` is mapped to 1 too, since "unlimited" without a concrete
# value is a footgun for a script that fans out subprocess workers.
test-reify: $(TARGET_RYSMITH) $(TARGET_RYLINK) $(TARGET_INTERP) $(TARGET_COMPILER)
	@JOBS=$$(echo "$(MAKEFLAGS)" | grep -oE -- '-j[0-9]+' | head -1 | sed 's/-j//'); \
	JOBS=$${JOBS:-1}; \
	$(PY) -m test.lib.run_reify_diff_tests --rysmith ./$(TARGET_RYSMITH) --symiri ./$(TARGET_INTERP) --symirc ./$(TARGET_COMPILER) --rylink ./$(TARGET_RYLINK) -n 100 --seed 1234 -j $$JOBS

# Aggregator. Recursive $(MAKE) calls keep per-component logs separated and
# let CI selectively re-run a single group on retry.
test: $(TARGET_INTERP) $(TARGET_COMPILER) $(TARGET_SOLVER) $(TARGET_RYSMITH) $(TARGET_RYLINK)
	$(MAKE) test-unit
	$(MAKE) test-frontend
	$(MAKE) test-interp
	$(MAKE) test-backends
	$(MAKE) cross-validation
	$(MAKE) test-solver
	$(MAKE) test-reify
