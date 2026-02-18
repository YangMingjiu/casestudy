CC     = mpicc
CFLAGS = -O2 -Wall
LIBS   = -llapack -lblas -lm

# ---------------------------------------------------------------
# Build
# ---------------------------------------------------------------
all: tsqr

tsqr: tsqr.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------
# Question 2: correctness test (default m=1200, n=6)
# ---------------------------------------------------------------
run: tsqr
	mpirun -np 4 ./tsqr

# ---------------------------------------------------------------
# Question 3: scaling benchmark + plot
# ---------------------------------------------------------------
scale: tsqr
	@echo "Running scaling benchmarks..."
	@bash run_scaling.sh
	@echo "Generating plots..."
	@python3 plot_scaling.py

# ---------------------------------------------------------------
clean:
	rm -f tsqr results_vary_m.csv results_vary_n.csv tsqr_scaling.png
