#include <stdio.h>
#include <algorithm>
#include <pthread.h>
#include <math.h>
#include <immintrin.h>

#include "CycleTimer.h"
#include "sqrt_ispc.h"

using namespace ispc;

extern void sqrtSerial(int N, float startGuess, float *values, float *output);

static void verifyResult(int N, float *result, float *gold)
{
    for (int i = 0; i < N; i++)
    {
        if (fabs(result[i] - gold[i]) > 1e-4)
        {
            printf("Error: [%d] Got %f expected %f\n", i, result[i], gold[i]);
        }
    }
}

void sqrtAVX2(int N, float *values, float *output)
{
    __m256 vec_threshhold = _mm256_set1_ps(0.00001f);
    for (int i = 0; i < N; i += 8)
    {
        __m256 vec_values = _mm256_loadu_ps(&values[i]);
        __m256 vec_guess = _mm256_set1_ps(1.0f);

        __m256 vec_errors = _mm256_mul_ps(vec_guess, vec_guess);
        vec_errors = _mm256_mul_ps(vec_errors, vec_values);
        vec_errors = _mm256_sub_ps(vec_errors, _mm256_set1_ps(1.0f));
        vec_errors = _mm256_and_ps(vec_errors, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));

        while (1)
        {
            __m256 mask = _mm256_cmp_ps(vec_errors, vec_threshhold, _CMP_GT_OQ);
            if (_mm256_testz_ps(mask, mask))
                break;
            __m256 a = _mm256_mul_ps(vec_guess, _mm256_set1_ps(3.f));
            __m256 b = _mm256_mul_ps(vec_guess, _mm256_mul_ps(vec_guess, _mm256_mul_ps(vec_guess, vec_values)));
            __m256 temp_guess = _mm256_sub_ps(a, b);
            temp_guess = _mm256_mul_ps(temp_guess, _mm256_set1_ps(0.5f));
            __m256 temp_errors = _mm256_mul_ps(temp_guess, _mm256_mul_ps(temp_guess, vec_values));
            temp_errors = _mm256_sub_ps(temp_errors, _mm256_set1_ps(1.0f));
            temp_errors = _mm256_and_ps(temp_errors, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
            vec_guess = _mm256_blendv_ps(vec_guess, temp_guess, mask);
            vec_errors = _mm256_blendv_ps(vec_errors, temp_errors, mask);
        }

        _mm256_storeu_ps(&output[i], _mm256_mul_ps(vec_guess, vec_values));
    }
}

int main()
{

    const unsigned int N = 20 * 1000 * 1000;
    const float initialGuess = 1.0f;

    float *values = new float[N];
    float *output = new float[N];
    float *gold = new float[N];

    for (unsigned int i = 0; i < N; i++)
    {
        // TODO: CS149 students.  Attempt to change the values in the
        // array here to meet the instructions in the handout: we want
        // to you generate best and worse-case speedups

        // starter code populates array with random input values
        values[i] = .001f + 2.998f * static_cast<float>(rand()) / RAND_MAX;

        // maximum speedup
        // values[i] = 2.999f;

        // minimum speedup
        // if (i % 8 == 0)
        //     values[i] = 2.999f;
        // else
        //     values[i] = 1.0f;
    }

    // generate a gold version to check results
    for (unsigned int i = 0; i < N; i++)
        gold[i] = sqrt(values[i]);

    //
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i)
    {
        double startTime = CycleTimer::currentSeconds();
        sqrtSerial(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[sqrt serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    verifyResult(N, output, gold);

    //
    // Compute the image using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minISPC = 1e30;
    for (int i = 0; i < 3; ++i)
    {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minISPC = std::min(minISPC, endTime - startTime);
    }

    printf("[sqrt ispc]:\t\t[%.3f] ms\n", minISPC * 1000);

    verifyResult(N, output, gold);

    //
    // Compute the image using the manually implemented avx2 implementation; report the minimum
    // time of three runs.
    //
    double minAVX = 1e30;
    for (int i = 0; i < 3; ++i)
    {
        double startTime = CycleTimer::currentSeconds();
        sqrtAVX2(N, values, output);
        double endTime = CycleTimer::currentSeconds();
        minAVX = std::min(minAVX, endTime - startTime);
    }

    printf("[sqrt avx]:\t\t[%.3f] ms\n", minAVX * 1000);

    verifyResult(N, output, gold);

    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;

    //
    // Tasking version of the ISPC code
    //
    double minTaskISPC = 1e30;
    for (int i = 0; i < 3; ++i)
    {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc_withtasks(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minTaskISPC = std::min(minTaskISPC, endTime - startTime);
    }

    printf("[sqrt task ispc]:\t[%.3f] ms\n", minTaskISPC * 1000);

    verifyResult(N, output, gold);

    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial / minISPC);
    printf("\t\t\t\t(%.2fx speedup from AVX)\n", minSerial / minAVX);
    printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial / minTaskISPC);

    delete[] values;
    delete[] output;
    delete[] gold;

    return 0;
}
