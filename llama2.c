/*
Please download the model and tokenizer to the same folder:
$ wget -O model.bin https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin
$ wget -O tokenizer.bin https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin
*/

#define _GNU_SOURCE // keep this line
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "utilities.h"

// YOUR CODE STARTS HERE
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

// global variables
#define MAX_THREAD_NUMBER 128
struct rusage main_usage; // get usage for main thread
struct thread_args
{
    float *out;
    float *vec;
    float *mat;
    int col;
    int row;
    int threadID;
};

sem_t main_semaphore;
sem_t mutex;                      // binary semaphore for mutex lock
sem_t workers[MAX_THREAD_NUMBER]; // array of semaphores, 1 for each worker

int count;     // number of threads that have finished
int n;         // total number of threads
int terminate; // flag to terminate threads

pthread_t threads[MAX_THREAD_NUMBER];
struct thread_args args[MAX_THREAD_NUMBER];

struct rusage thread_usage[MAX_THREAD_NUMBER]; // get usage for each thread
struct rusage main_usage;                      // get usage for main thread

void *thr_func(void *arg)
{
    while (1)
    {
        // wait for signal from main thread to start
        sem_wait(&workers[((struct thread_args *)arg)->threadID]);

        // check if its going to terminate
        if (terminate)
        {
            getrusage(RUSAGE_THREAD, &thread_usage[((struct thread_args *)arg)->threadID]); // store thread system usage
            pthread_exit(NULL);                                                             // terminate thread
        }

        // get arguments
        struct thread_args *args = (struct thread_args *)arg;
        float *out = args->out;
        float *vec = args->vec;
        float *mat = args->mat;
        int col = args->col;
        int row = args->row;
        int k = args->threadID;

        // calculate start to end row
        int start_row;
        int end_row;
        if (row % n == 0) // row is evenly divided by number of threads
        {
            start_row = k * (row / n);
            end_row = (k + 1) * (row / n) - 1;
        }
        else // row is not evenly divided by number of threads
        {
            start_row = k * ceil(row / n);
            if (k != n - 1)
            {
                end_row = (k + 1) * ceil(row / n) - 1;
            }
            else
            {
                end_row = row - 1; // last thread handles the remaining of the rows
            }
        }

        // perform computation
        for (int i = start_row; i <= end_row; i++)
        {
            float val = 0.0f;
            for (int j = 0; j < col; j++)
            {
                val += mat[i * col + j] * vec[j];
            }
            out[i] = val;
        }

        sem_wait(&mutex);
        count++;

        if (count == n)
        {
            sem_post(&main_semaphore); // signal main thread if last thread has finished
        }
        sem_post(&mutex);
    }
}

/**
 * @brief
 *
 * @param out
 * @param vec
 * @param mat
 * @param col
 * @param row
 */
void mat_vec_mul(float *out, float *vec, float *mat, int col, int row)
{
    // set parameters
    for (int i = 0; i < n; i++)
    {
        args[i].out = out;
        args[i].vec = vec;
        args[i].mat = mat;
        args[i].col = col;
        args[i].row = row;
    }
    count = 0;

    // signal all threads to start
    for (int i = 0; i < n; i++)
    {
        sem_post(&workers[i]);
    }

    // wait for all threads to finish
    sem_wait(&main_semaphore);
}

/**
 * @brief create n threads and initialize necessary variables
 *
 * @param thr_count
 * @return int
 */
int create_mat_vec_mul(int thr_count)
{
    // initialize necessary variables
    sem_init(&main_semaphore, 0, 0);
    sem_init(&mutex, 0, 1);
    for (int i = 0; i < thr_count; i++)
    {
        sem_init(&workers[i], 0, 0); // create the semaphores for each thread
    }

    count = 0;
    n = thr_count;
    terminate = 0;

    // create n threads, each with unique ID
    for (int i = 0; i < thr_count; i++)
    {
        args[i].threadID = i;
        pthread_create(&threads[i], NULL, thr_func, (void *)&args[i]);
    }
}

int destroy_mat_vec_mul()
{
    terminate = 1;
    for (int i = 0; i < n; i++)
    {
        sem_post(&workers[i]); // signal all threads to terminate
    }
    for (int i = 0; i < n; i++)
    {
        pthread_join(threads[i], NULL); // wait for all threads to terminate
    }

    // prints each thread's system usage
    for (int i = 0; i < n; i++)
    {
        printf("Thread %d has completed - user: %.4f s, system: %.4f s\n",
               i,
               (thread_usage[i].ru_utime.tv_sec + thread_usage[i].ru_utime.tv_usec / 1000000.0),
               (thread_usage[i].ru_stime.tv_sec + thread_usage[i].ru_stime.tv_usec / 1000000.0));
    }

    // prints the program's system usage
    getrusage(RUSAGE_SELF, &main_usage);
    printf("main thread - user: %.4f s, system: %.4f s\n",
           (main_usage.ru_utime.tv_sec + main_usage.ru_utime.tv_usec / 1000000.0),
           (main_usage.ru_stime.tv_sec + main_usage.ru_stime.tv_usec / 1000000.0));

    // cleans up resources allocated for multi-threading
    sem_destroy(&main_semaphore);
    sem_destroy(&mutex);
    for (int i = 0; i < n; i++)
    {
        sem_destroy(&workers[i]);
    }
}

// YOUR CODE ENDS HERE

int transformer(int token, int pos, LLMConfig *p, LLMRuntime *s, LLMWeight *w)
{

    // ar few convenience variables
    int dim = p->dim, hidden_dim = p->hidden_dim, head_size = p->dim / p->n_heads;

    // copy the token embedding into x
    memcpy(s->x, &(w->token_embedding_table[token * dim]), dim * sizeof(float));

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++)
    {

        // Attention
        {
            // attention normalization
            normalize(s->xb, s->x, w->rms_att_weight + l * dim, dim);

            // q, k, v = w_q @ x, w_k @ x, w_v @ x, respectively
            mat_vec_mul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
            mat_vec_mul(s->k, s->xb, w->wk + l * dim * dim, dim, dim);
            mat_vec_mul(s->v, s->xb, w->wv + l * dim * dim, dim, dim);

            // apply positional embedding
            position_embedding(s->q, s->k, w, pos, p->dim, p->n_heads);

            // save intermediate result for later reference
            key_value_cache(l, pos, p, s);

            // attention calculation
            attention(l, pos, p, s, w);

            // wo @ x to get final result
            mat_vec_mul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

            // residual connection back into x
            accum(s->x, s->xb2, dim);
        }

        // Feed-Forward Network: w2 @ (silu(w1 @ x) * (w3 @ x)), * is element-wise multiply
        {
            // FFN Normalization
            normalize(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);

            // w1 @ x
            mat_vec_mul(s->h1, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
            // silu(w1 @ x)
            silu(s->h1, hidden_dim);
            // w3 @ x
            mat_vec_mul(s->h2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
            // silu(w1 @ x) * (w3 @ x)
            element_wise_mul(s->h1, s->h2, hidden_dim);
            // w2 @ (silu(w1 @ x) * (w3 @ x))
            mat_vec_mul(s->xb, s->h1, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

            // residual connection
            accum(s->x, s->xb, dim);
        }
    }

    // final normalization
    normalize(s->x, s->x, w->rms_final_weight, dim);
    // classifier into logits
    mat_vec_mul(s->logits, s->x, w->token_embedding_table, p->dim, p->vocab_size);
    // apply the temperature to the logits
    for (int q = 0; q < p->vocab_size; q++)
    {
        s->logits[q] /= 0.9f;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(s->logits, p->vocab_size);
    // now sample from this distribution to get the next token
    return sample(s->logits, p->vocab_size);
}

int main(int argc, char *argv[])
{

    unsigned int seed;
    int thr_count;

    if (argc == 3)
    {
        seed = atoi(argv[1]);
        thr_count = atoi(argv[2]);
    }
    else
    {
        printf("Usage: ./compiled <seed> <thr_count>\n");
        return 1;
    }

    // Initialize
    srand(seed);
    create_mat_vec_mul(thr_count);

    // load model
    LLMConfig config;
    LLMWeight weights;
    if (load_LLM_Config_Weight(&config, &weights) == 1)
    {
        return 1;
    }

    // load tokenizer
    char **vocab = malloc(config.vocab_size * sizeof(char *));
    if (load_tokenizer(vocab, config.vocab_size) == 1)
    {
        return 1;
    }

    // create and init the application LLMRuntime
    LLMRuntime state;
    malloc_LLMRuntime(&state, &config);

    // the current position we are in
    long start = time_in_ms();

    int next, token = 1, pos = 0; // token = 1 -> <START>
    while (pos < config.seq_len)
    {

        // forward the transformer to get logits for the next token
        next = transformer(token, pos, &config, &state, &weights);

        printf("%s", vocab[next]);
        fflush(stdout); // force print

        token = next;
        pos++;
    }

    long end = time_in_ms();
    printf("\n\nlength: %d, time: %f s, achieved tok/s: %f\n", config.seq_len, (double)(end - start) / 1000, config.seq_len / (double)(end - start) * 1000);

    // cleanup
    destroy_mat_vec_mul();
    free_LLMRuntime(&state);
    free_LLMWeight(&weights);
    for (int i = 0; i < config.vocab_size; i++)
    {
        free(vocab[i]);
    }
    free(vocab);
    return 0;
}
