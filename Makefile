#================================================================
#                     TRACMASS MAKEFILE
#================================================================

# Project and case definition

PROJECT	          = GLORYS12_ORCA
CASE              = Atlantic_gates_Lom_hf

RUNFILE 	      = runtracmass
ARCH              = cyclone

NETCDFLIBS        = automatic-44
NCDF_ROOT         = ${HOME}/local/netcdf-fortran/4.5.3
#================================================================

# Possible architectures:
# tetralith    (Swedish HPC with intel)

# Possible netCDF settings:
# automatic    (set by nc-config)
# automatic-44 (set by nf-config, for netCDF version >4.4)
# custom       (custom netCDF library, need to specify NCDF_ROOT)
# none         (no netCDF library)

#================================================================
# ***************************************************************
#================================================================

# Read the project Makefile
PROJMAKE           := $(wildcard projects/$(PROJECT)/Makefile.prj)
CASEMAKE           := $(wildcard projects/$(PROJECT)/Makefile.prj)

ifneq ($(strip $(CASEMAKE)),)
include projects/$(PROJECT)/Makefile.prj
else
ifneq ($(strip $(PROJMAKE)),)
include projects/$(PROJECT)/Makefile.prj
endif
endif

PROJECT_FLAG      = -DPROJECT_NAME=\'$(PROJECT)\'
CASE_FLAG         = -DCASE_NAME=\'$(CASE)\'

#================================================================

# NetCDF libraries
ifeq ($(NETCDFLIBS),none)
LIB_DIR =
INC_DIR =
ORM_FLAGS += -Dno_netcdf

else ifeq ($(NETCDFLIBS),automatic)
LIB_DIR = $(shell nc-config --flibs)
INC_DIR = -I$(shell nc-config --includedir)

else ifeq ($(NETCDFLIBS),automatic-44)
LIB_DIR = $(shell nf-config --flibs)
INC_DIR = $(shell nf-config --cflags)

else ifeq ($(NETCDFLIBS),custom)
LIB_DIR = -L${HOME}/local/netcdf-fortran/4.5.3/lib -L${HOME}/local/netcdf/4.8.1/lib -lnetcdf -lnetcdff
INC_DIR = -I${HOME}/local/netcdf-fortran/4.5.3/include

else
NCDF_ROOT = /usr

LIB_DIR = -L$(NCDF_ROOT)/lib -lnetcdf -lnetcdff
INC_DIR	= -I$(NCDF_ROOT)/include

endif

# Fortran compiler and flags
ifeq ($(ARCH),tetralith)
FC = ifort
FF = -g -O3 -traceback -pg

else ifeq ($(ARCH),nird)
FC = gfortran
FF = -g -O3 -fbacktrace -fbounds-check -Wall -Wno-maybe-uninitialized -Wno-unused-dummy-argument

else
FC = gfortran
FF = -g -O3 -fbacktrace -fbounds-check -Wall -Wno-maybe-uninitialized -Wno-unused-dummy-argument

endif

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NCDF_ROOT}/lib

# Path to sources
VPATH = src:projects/$(PROJECT):_build

all: runfile

ifneq ($(strip $(CASE)),)
	cp projects/$(PROJECT)/namelist_$(CASE).in namelist.in
else
ifneq ($(strip $(PROJECT)),)
	cp projects/$(PROJECT)/namelist_$(PROJECT).in namelist.in
endif
endif


#================================================================

# Object definitions
OBJDIR := _build

objects := $(addprefix $(OBJDIR)/,mod_vars.o mod_subdomain.o mod_getfile.o mod_calendar.o \
	mod_tracerf.o mod_tracers.o setup_grid.o kill_zones.o mod_vertvel.o mod_swap.o read_field.o mod_clock.o  \
	mod_write.o mod_error.o mod_seed.o mod_diffusion.o mod_divergence.o mod_stream.o \
	mod_pos_tstep.o mod_pos_tanalytical.o mod_init.o mod_print.o mod_loop.o mod_postprocess.o TRACMASS.o)

$(OBJDIR)/%.o : %.F90
		$(FC) $(FF) -c $(ORM_FLAGS) $(PROJECT_FLAG) $(CASE_FLAG) $(INC_DIR) $(LIB_DIR) $< -o $@

$(objects) : | $(OBJDIR)

$(OBJDIR):
			mkdir -p $(OBJDIR)

#================================================================

runfile : $(objects)

	$(FC) $(FF) $(ORM_FLAGS) -o $(RUNFILE) $(objects) $(INC_DIR) $(LIB_DIR) -Wl,-rpath=${NCDF_ROOT}/lib,-rpath=${HOME}/local/netcdf/4.8.1/lib

test :

	sed '42s~.*~NCDIR="$(LIB_DIR) $(INC_DIR)" ~' src/_funit/runtest.sh > runtest.sh
	chmod +x runtest.sh

.PHONY : clean

clean:

	-rm -rf *.o *.mod *.out *.dSYM *.csv fort.* *.x *.in
	-rm -rf _build
	test -s runtest.sh && rm runtest.sh || true
	test -s $(RUNFILE) && rm $(RUNFILE) || true

.PHONY : help

help :
	@echo
	@echo "make       : Generate TRACMASS runfile '$(RUNFILE).x'."
	@echo "make test  : Generate test-suite runscripts 'runtest.sh'."
	@echo "make clean : Remove auto-generated files."
	@echo
