/*
 * kmeans_mpi.c
 *
 * Parallel K-Means using MPI (input partitioning approach from lecture).
 *
 * rank 0 reads the whole dataset, scatters rows to each process, then
 * each process handles its own chunk every iteration. We use Allreduce
 * to sum up partial counts/sums so every rank can update the centers,
 * then Gatherv at the end to collect all assignments back to rank 0.
 *
 * ndata might not divide evenly by nprocs so we use Scatterv/Gatherv
 * with per-rank counts and displacements instead of the basic versions.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

/* read the file once just to count lines and figure out dimensionality
 * from the first line (everything after the colon) */
int count_lines_tokens(const char *filename, int *dim) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("fopen"); exit(1); }

    char line[65536];
    int n = 0;
    *dim = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (n == 0) {
            char *p = strchr(line, ':');
            if (!p) { fprintf(stderr, "Bad data format\n"); exit(1); }
            int d = 0;
            char *tok = strtok(p + 1, " \t\n");
            while (tok) { d++; tok = strtok(NULL, " \t\n"); }
            *dim = d;
        }
        n++;
    }
    fclose(fp);
    return n;
}

/* load labels and feature vectors into flat arrays.
 * features is stored row-major: features[i*dim + d] = feature d of sample i */
void load_data(const char *filename, int ndata, int dim,
               float *features, int *labels, int *nlabels) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("fopen"); exit(1); }

    char line[65536];
    int i = 0, maxlabel = -1;

    while (fgets(line, sizeof(line), fp)) {
        char *ptr = strchr(line, ':');
        if (!ptr) { fprintf(stderr, "Bad data format\n"); exit(1); }
        *ptr = '\0';
        int label = atoi(line);
        if (label > maxlabel) maxlabel = label;
        labels[i] = label;

        char *tok = strtok(ptr + 1, " \t\n");
        for (int d = 0; d < dim && tok; d++, tok = strtok(NULL, " \t\n"))
            features[i * dim + d] = atof(tok);
        i++;
    }
    fclose(fp);
    *nlabels = maxlabel + 1;
}

/* compute cluster centers as the mean of all points assigned to each cluster */
void compute_centers(float *features, int *assigns, int ndata, int dim,
                     int nclust, float *centers, int *counts) {
    memset(centers, 0, nclust * dim * sizeof(float));
    memset(counts,  0, nclust * sizeof(int));

    for (int i = 0; i < ndata; i++) {
        int c = assigns[i];
        counts[c]++;
        for (int d = 0; d < dim; d++)
            centers[c * dim + d] += features[i * dim + d];
    }

    for (int c = 0; c < nclust; c++)
        if (counts[c] > 0)
            for (int d = 0; d < dim; d++)
                centers[c * dim + d] /= counts[c];
}

/* print confusion matrix showing how cluster assignments line up with true labels */
void print_confusion(int *assigns, int *labels, int ndata,
                     int nclust, int nlabels) {
    int *mat     = calloc(nlabels * nclust, sizeof(int));
    int *row_tot = calloc(nlabels,          sizeof(int));
    int *col_tot = calloc(nclust,           sizeof(int));

    for (int i = 0; i < ndata; i++) {
        int r = labels[i], c = assigns[i];
        mat[r * nclust + c]++;
        row_tot[r]++;
        col_tot[c]++;
    }

    printf("==CONFUSION MATRIX + COUNTS==\n");
    printf("LABEL \\ CLUST\n");
    printf("   ");
    for (int c = 0; c < nclust; c++) printf("%5d", c);
    printf("  TOT\n");

    for (int r = 0; r < nlabels; r++) {
        printf(" %2d:", r);
        for (int c = 0; c < nclust; c++) printf("%5d", mat[r * nclust + c]);
        printf("%5d\n", row_tot[r]);
    }

    printf("TOT");
    for (int c = 0; c < nclust; c++) printf("%5d", col_tot[c]);
    printf("%5d\n", ndata);
    printf("\n");

    free(mat); free(row_tot); free(col_tot);
}

/* save each cluster center as a PGM image - only works when dim is a perfect
 * square (e.g. 784 = 28x28 for MNIST), otherwise just skip it */
void save_pgm_files(float *centers, int nclust, int dim, const char *savedir) {
    int dim_root = (int)sqrt((double)dim);
    if (dim_root * dim_root != dim) return;

    float maxfeat = 0.0f;
    for (int i = 0; i < nclust * dim; i++)
        if (centers[i] > maxfeat) maxfeat = centers[i];

    for (int c = 0; c < nclust; c++) {
        char fname[256];
        sprintf(fname, "%s/cent_%04d.pgm", savedir, c);
        if (c == 0) printf("Saving cluster centers to %s ...\n", fname);

        FILE *fp = fopen(fname, "w");
        if (!fp) { perror("fopen"); continue; }

        fprintf(fp, "P2\n%d %d\n%d\n", dim_root, dim_root, (int)(maxfeat + 0.5f));
        for (int d = 0; d < dim; d++) {
            if (d > 0 && d % dim_root == 0) fprintf(fp, "\n");
            fprintf(fp, "%3.0f ", centers[c * dim + d]);
        }
        fprintf(fp, "\n");
        fclose(fp);
    }
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr, "usage: %s <datafile> <nclust> [savedir] [maxiter]\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    char *datafile = argv[1];
    int   nclust   = atoi(argv[2]);
    char *savedir  = (argc > 3) ? argv[3] : ".";
    int   MAXITER  = (argc > 4) ? atoi(argv[4]) : 100;

    /* only rank 0 reads the file */
    int    ndata = 0, dim = 0, nlabels = 0;
    float *all_features = NULL;
    int   *all_labels   = NULL;
    int   *all_assigns  = NULL;

    if (rank == 0) {
        mkdir(savedir, 0777);
        ndata        = count_lines_tokens(datafile, &dim);
        all_features = malloc((size_t)ndata * dim * sizeof(float));
        all_labels   = malloc((size_t)ndata       * sizeof(int));
        all_assigns  = malloc((size_t)ndata       * sizeof(int));
        load_data(datafile, ndata, dim, all_features, all_labels, &nlabels);

        printf("datafile: %s\n", datafile);
        printf("nclust: %d\n",   nclust);
        printf("savedir: %s\n",  savedir);
        printf("ndata: %d\n",    ndata);
        printf("dim: %d\n\n",    dim);
    }

    /* broadcast ndata/dim/nlabels so all ranks can allocate their local buffers */
    MPI_Bcast(&ndata,   1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim,     1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nlabels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* figure out how many rows each rank gets. the first (ndata % nprocs)
     * ranks get one extra row so nothing is left out */
    int *row_counts = malloc(nprocs * sizeof(int));
    int *row_displs = malloc(nprocs * sizeof(int));
    int *flt_counts = malloc(nprocs * sizeof(int)); /* same but multiplied by dim */
    int *flt_displs = malloc(nprocs * sizeof(int));

    int base = ndata / nprocs, extra = ndata % nprocs;
    for (int r = 0; r < nprocs; r++) {
        row_counts[r] = base + (r < extra ? 1 : 0);
        row_displs[r] = (r == 0) ? 0 : row_displs[r-1] + row_counts[r-1];
        flt_counts[r] = row_counts[r] * dim;
        flt_displs[r] = row_displs[r] * dim;
    }

    int    local_n       = row_counts[rank];
    float *local_feat    = malloc((size_t)local_n * dim * sizeof(float));
    int   *local_assigns = malloc((size_t)local_n * sizeof(int));

    /* scatter rows of the feature matrix to each rank.
     * using Scatterv instead of Scatter because chunks may not be equal size */
    MPI_Scatterv(all_features, flt_counts, flt_displs, MPI_FLOAT,
                 local_feat, local_n * dim,             MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    /* rank 0 seeds initial centers using round-robin assignment, then
     * broadcasts the centers so everyone starts from the same point */
    float *centers    = malloc((size_t)nclust * dim * sizeof(float));
    int   *tmp_counts = malloc(nclust * sizeof(int));

    if (rank == 0) {
        for (int i = 0; i < ndata; i++)
            all_assigns[i] = i % nclust;
        compute_centers(all_features, all_assigns, ndata, dim, nclust,
                        centers, tmp_counts);
    }

    MPI_Bcast(centers, nclust * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    free(tmp_counts);

    /* each rank seeds its local assignments consistently with what rank 0 did */
    int offset = row_displs[rank];
    for (int i = 0; i < local_n; i++)
        local_assigns[i] = (offset + i) % nclust;

    float *local_sums    = malloc((size_t)nclust * dim * sizeof(float));
    float *global_sums   = malloc((size_t)nclust * dim * sizeof(float));
    int   *local_counts  = malloc(nclust * sizeof(int));
    int   *global_counts = malloc(nclust * sizeof(int));

    if (rank == 0) {
        printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
        printf("ITER NCHANGE CLUST_COUNTS\n");
    }

    int curiter = 1;
    int nchanges = ndata; /* set to ndata so we always run at least one iteration */

    while (nchanges > 0 && curiter <= MAXITER) {

        /* step 1: each rank computes partial sums/counts for its local rows */
        memset(local_sums,   0, (size_t)nclust * dim * sizeof(float));
        memset(local_counts, 0, nclust * sizeof(int));

        for (int i = 0; i < local_n; i++) {
            int c = local_assigns[i];
            local_counts[c]++;
            for (int d = 0; d < dim; d++)
                local_sums[c * dim + d] += local_feat[i * dim + d];
        }

        /* step 2: sum everything across ranks so we get global counts/sums.
         * Allreduce instead of Reduce because every rank needs the result
         * to update its own copy of the centers */
        MPI_Allreduce(local_counts, global_counts, nclust,
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_sums, global_sums, nclust * dim,
                      MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        /* update centers: divide summed features by how many points landed there */
        for (int c = 0; c < nclust; c++)
            if (global_counts[c] > 0)
                for (int d = 0; d < dim; d++)
                    centers[c * dim + d] = global_sums[c * dim + d] / global_counts[c];

        /* step 3: re-assign each local point to its nearest center */
        memset(local_counts, 0, nclust * sizeof(int));
        int local_changes = 0;

        for (int i = 0; i < local_n; i++) {
            float best_dist = INFINITY;
            int   best_c    = 0;

            for (int c = 0; c < nclust; c++) {
                float dist = 0.0f;
                for (int d = 0; d < dim; d++) {
                    float diff = local_feat[i * dim + d] - centers[c * dim + d];
                    dist += diff * diff;
                }
                if (dist < best_dist) { best_dist = dist; best_c = c; }
            }

            if (best_c != local_assigns[i]) {
                local_assigns[i] = best_c;
                local_changes++;
            }
            local_counts[best_c]++;
        }

        /* step 4: sum up changes across all ranks to check convergence */
        MPI_Allreduce(&local_changes, &nchanges, 1,
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts, global_counts, nclust,
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("%3d: %6d |", curiter, nchanges);
            for (int c = 0; c < nclust; c++) printf(" %4d", global_counts[c]);
            printf("\n");
        }
        curiter++;
    }

    if (rank == 0) {
        if (curiter > MAXITER)
            printf("WARNING: maximum iteration %d exceeded, may not have converged\n", MAXITER);
        else
            printf("CONVERGED: after %d iterations\n", curiter);
    }

    /* collect all local assignment arrays back to rank 0 using Gatherv
     * (inverse of the Scatterv we did at the start, same counts/displs) */
    MPI_Gatherv(local_assigns, local_n, MPI_INT,
                all_assigns, row_counts, row_displs, MPI_INT,
                0, MPI_COMM_WORLD);

    /* rank 0 handles all output */
    if (rank == 0) {
        printf("\n");
        print_confusion(all_assigns, all_labels, ndata, nclust, nlabels);

        char outfile[256];
        sprintf(outfile, "%s/labels.txt", savedir);
        printf("Saving cluster labels to file %s\n", outfile);

        FILE *fp = fopen(outfile, "w");
        for (int i = 0; i < ndata; i++)
            fprintf(fp, "%2d %2d\n", all_labels[i], all_assigns[i]);
        fclose(fp);

        save_pgm_files(centers, nclust, dim, savedir);

        free(all_features);
        free(all_labels);
        free(all_assigns);
    }

    free(local_feat);    free(local_assigns);
    free(centers);
    free(local_sums);    free(global_sums);
    free(local_counts);  free(global_counts);
    free(row_counts);    free(row_displs);
    free(flt_counts);    free(flt_displs);

    MPI_Finalize();
    return 0;
}
