/*  
 *  sw_par
 *
 *  Copyright April 2019 by Nikolaos Alachiotis
 *
 *  This program is free software; you may redistribute it and/or modify its
 *  under the terms of the GNU General Public License as published by the Free
 *  Software Foundation; either version 2 of the License, or (at your option)
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 *  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 *  for more details.
 *
 *  For any other enquiries send an email to
 *  Nikolaos Alachiotis (n.alachiotis@gmail.com)
 *  
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <limits.h>
#include <ctype.h>
#include <stdbool.h>


#ifdef _SHOW_RESULTS
#define PRINT_MATRIX 	
#define QSIZE 100	
#define DSIZE 10000
#define ITERATIONS 10	
#else
#define QSIZE 20000
#define DSIZE 100000
#define ITERATIONS 1
#endif

#ifdef _SEQUENTIAL
#ifdef _MODE_1
#define REFERENCE
#elif _MODE_2
#define ANTIDIAGONALS
#elif _MODE_3
#define ANTIDIAGONALS_WITH_MLT
#elif _MODE_4
#define ANTIDIAGONALS_WITH_MLT
#define ANTIDIAGONALS_WITH_MLT_NO_IF
#elif _MODE_5
#define REFERENCE
#define UNROLL4
#elif _MODE_6
#define REFERENCE
#define UNROLL4
#define JAM
#elif _MODE_7
#define ANTIDIAGONALS_WITH_MLT
#define ANTIDIAGONALS_WITH_MLT_NO_IF
#define UNROLL4
#elif _MODE_8
#define ANTIDIAGONALS_WITH_MLT
#define ANTIDIAGONALS_WITH_MLT_NO_IF
#define UNROLL4
#define JAM
#endif
#else
#ifdef _PARALLEL
#define ANTIDIAGONALS_WITH_MLT
#define ANTIDIAGONALS_WITH_MLT_NO_IF
#define PTHREADS
#define THREADS 4
//#define PINNING
#ifdef _MODE_1
#define PTHREAD_BARRIER
#elif _MODE_2
// array-based counting barrier
#elif _MODE_3
#define SENSE_REVERSAL_BARRIER
#elif _MODE_4
#define SENSE_REVERSAL_BARRIER
#define UNROLL2
#elif _MODE_5
#define SENSE_REVERSAL_BARRIER
#define UNROLL2
#define JAM
#endif
#endif
#endif


double gettime(void);
int getMax (int a, int b, int c);

double gettime(void)
{
	struct timeval ttime;
	gettimeofday(&ttime , NULL);
	return ttime.tv_sec + ttime.tv_usec * 0.000001;
}

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MAX3(a,b,c) MAX(0, MAX(a,MAX(b,c)))

#ifdef PTHREADS
#include <pthread.h>
#include <unistd.h>

#define EXIT 127
#define BUSYWAIT 0
#define COMPUTE_ANTIDIAGONAL 1
#define COMPUTE_ANTIDIAGONAL_P2_A 2
#define COMPUTE_ANTIDIAGONAL_P2_B 3
#define COMPUTE_ANTIDIAGONAL_P3 4

static pthread_t * workerThread; 
//int lockVal;

#ifdef PTHREAD_BARRIER
static pthread_barrier_t barrier;
#endif

#ifdef SENSE_REVERSAL_BARRIER
static int count;
static bool sense;

typedef struct localsense_t {
	bool lsense;
}localsense_t;

static localsense_t *localsense_list=NULL;
void sense_reversal_barrier_init (int num_threads);
void sense_reversal_barrier (int tid, int num_threads);
void sense_reversal_barrier_destroy (void);

void sense_reversal_barrier_init (int num_threads)
{
	int i;
	sense = true;
	count = num_threads;
	if (localsense_list==NULL) 
	{
		localsense_list = (localsense_t *)malloc(sizeof(localsense_t)*(unsigned long)num_threads);
		assert(localsense_list!=NULL);
	}
	for (i=0; i<num_threads; i++) 
	{
		localsense_list[i].lsense=true;
	}
}


void sense_reversal_barrier (int tid, int num_threads) 
{
	int threadno = tid;
	localsense_list[threadno].lsense = !localsense_list[threadno].lsense;

	if (__sync_fetch_and_sub (&count, 1) == 1) {
		count = num_threads;
		sense = localsense_list[threadno].lsense;
	}
	else {
		while(sense != localsense_list[threadno].lsense) __sync_synchronize();
	}
}

void sense_reversal_barrier_destroy (void) 
{
	if(localsense_list!=NULL)
		free(localsense_list);

	localsense_list=NULL;
}
#endif

typedef struct
{
	int diagSize;
	int diagDoffset;
	int MATCH;
	int MISMATCH;
	int GAP;
	int cu;
	int maxVal;
	int diagIndex;		
	int jstart0;
	int Qindex;
	char * Q;
	char * D;
	int ** diag;
} antidiagonalData_t;


typedef struct 
{
	int threadID;
	int threadTOTAL;

	int threadBARRIER;
	int threadOPERATION;

	antidiagonalData_t antidiagonalData;	

} threadData_t;

void computeAntidiagonalPart (threadData_t * currentThread);
void initializeThreadData(threadData_t * cur, int i, int threads, int MATCH, int MISMATCH, int GAP, char * Q, char * D, int ** diag, int sizeQ);
void setThreadArguments_COMPUTE_ANTIDIAGONAL_operation (threadData_t * threadData, int tid, int diagSize, int diagDoffset, int diagIndex, int jstart0);
void updateThreadArguments_COMPUTE_ANTIDIAGONAL_operation (threadData_t * threadData, int diagSize, int diagDoffset, int diagIndex, int jstart0);
static inline void syncThreadsBARRIER(threadData_t * threadData);
static inline void computeAntidiagonal (threadData_t * threadData);
static inline void computeAntidiagonal_P2_A (threadData_t * threadData);
static inline void computeAntidiagonal_P2_B (threadData_t * threadData);
static inline void computeAntidiagonal_P3 (threadData_t * threadData);
static inline void execFunctionMaster(threadData_t * threadData, int operation);
void startThreadOperations(threadData_t * threadData, int operation);
void * thread (void * x);
void terminateWorkerThreads(pthread_t * workerThreadL, threadData_t * threadData);

void initializeThreadData(threadData_t * cur, int i, int threads, int MATCH, int MISMATCH, int GAP, char * Q, char * D, int ** diag, int sizeQ)
{
	cur->threadID=i;
	cur->threadTOTAL=threads;
	cur->threadBARRIER=0;
	cur->threadOPERATION=BUSYWAIT;
	cur->antidiagonalData.diagSize = 0;
	cur->antidiagonalData.diagDoffset = 0;
	cur->antidiagonalData.MATCH = MATCH;
	cur->antidiagonalData.MISMATCH = MISMATCH;
	cur->antidiagonalData.GAP = GAP;
	cur->antidiagonalData.cu = 0;
	cur->antidiagonalData.maxVal = 0;
	cur->antidiagonalData.Q = Q;
	cur->antidiagonalData.D = D;
	cur->antidiagonalData.diag = diag;
	cur->antidiagonalData.diagIndex = 0;
	cur->antidiagonalData.Qindex = sizeQ-1;
}


void setThreadArguments_COMPUTE_ANTIDIAGONAL_operation (threadData_t * threadData, int tid, int diagSize, int diagDoffset, int diagIndex, int jstart0)
{
	threadData[tid].antidiagonalData.diagSize=diagSize;
	threadData[tid].antidiagonalData.diagDoffset=diagDoffset;
	threadData[tid].antidiagonalData.diagIndex=diagIndex;
	threadData[tid].antidiagonalData.jstart0=jstart0;
}

void updateThreadArguments_COMPUTE_ANTIDIAGONAL_operation (threadData_t * threadData, int diagSize, int diagDoffset, int diagIndex, int jstart0)
{
	int threadIndex = 0;
	for(threadIndex=0;threadIndex<threadData->threadTOTAL;threadIndex++)
		setThreadArguments_COMPUTE_ANTIDIAGONAL_operation (threadData, threadIndex, diagSize, diagDoffset, diagIndex, jstart0);

}

/*void lock ()
{
	while(__sync_lock_test_and_set(&lockVal,1)==1);
}

void unlock()
{
	lockVal=0;
}*/

static inline void syncThreadsBARRIER(threadData_t * threadData)
{
	int i, threads = threadData[0].threadTOTAL, barrierS=0;

	threadData[0].threadOPERATION=BUSYWAIT;

#ifdef SENSE_REVERSAL_BARRIER
	sense_reversal_barrier(0, threads);
#else
#ifdef PTHREAD_BARRIER
 	pthread_barrier_wait(&barrier);
#else
	while(barrierS!=threads)
	{
		barrierS=0;
		for(i=0;i<threads;i++)
			barrierS += threadData[i].threadBARRIER;
	}

	for(i=0;i<threads;i++)
		threadData[i].threadBARRIER=0;
#endif
#endif
}

static inline void computeAntidiagonal (threadData_t * threadData)
{
	int threadID = threadData->threadID;
	int totalThreads = threadData->threadTOTAL;

	int diagSize = threadData->antidiagonalData.diagSize;
	int diagDoffset = threadData->antidiagonalData.diagDoffset;
	int MATCH = threadData->antidiagonalData.MATCH;
	int MISMATCH = threadData->antidiagonalData.MISMATCH;
	int GAP = threadData->antidiagonalData.GAP;
	char * Q = threadData->antidiagonalData.Q;
	char * D = threadData->antidiagonalData.D;
	int ** diag = threadData->antidiagonalData.diag;
	int i = threadData->antidiagonalData.diagIndex;
	int jstart0 = threadData->antidiagonalData.jstart0;
	int cu = 0;
	int maxVal = 0;
#ifdef JAM
	int maxVal1 = 0, maxVal2 = 0;
#endif

	int tasksPerThread = diagSize / totalThreads;

	int jstart = tasksPerThread*threadID;
	int jstop = tasksPerThread*threadID+tasksPerThread-1;

	if(threadID==0)
		jstart = jstart0;

	if(threadID==totalThreads-1)
		jstop = diagSize-1-1;
	
	int j;
#ifdef UNROLL2
	int unroll=2;
#ifndef JAM
	for(j=jstart;j<jstop-unroll;j=j+unroll)
	{
		int isMatch = Q[i-j-1]==D[diagDoffset+j-1]?MATCH:MISMATCH;

		diag[i][j] = MAX3(diag[i-2][j-1]+isMatch, diag[i-1][j-1]+GAP, diag[i-1][j]+GAP);
		cu++;
		maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;

		isMatch = Q[i-(j+1)-1]==D[diagDoffset+(j+1)-1]?MATCH:MISMATCH;

		diag[i][(j+1)] = MAX3(diag[i-2][(j+1)-1]+isMatch, diag[i-1][(j+1)-1]+GAP, diag[i-1][(j+1)]+GAP);
		cu++;
		maxVal = diag[i][(j+1)]>=maxVal?diag[i][(j+1)]:maxVal;
	}
#else
	for(j=jstart;j<jstop-unroll;j=j+unroll)
	{
		int isMatch1 = Q[i-j-1]==D[diagDoffset+j-1]?MATCH:MISMATCH;
		int isMatch2 = Q[i-(j+1)-1]==D[diagDoffset+(j+1)-1]?MATCH:MISMATCH;

		diag[i][j] = MAX3(diag[i-2][j-1]+isMatch1, diag[i-1][j-1]+GAP, diag[i-1][j]+GAP);
		diag[i][(j+1)] = MAX3(diag[i-2][(j+1)-1]+isMatch2, diag[i-1][(j+1)-1]+GAP, diag[i-1][(j+1)]+GAP);

		cu=cu+2;
		maxVal1 = diag[i][j]>=maxVal1?diag[i][j]:maxVal1;
		maxVal2 = diag[i][(j+1)]>=maxVal2?diag[i][(j+1)]:maxVal2;
	}
	maxVal = maxVal1>=maxVal2?maxVal1:maxVal2;
#endif
	for(;j<=jstop;j++)
	{
		int isMatch = Q[i-j-1]==D[diagDoffset+j-1]?MATCH:MISMATCH;

		diag[i][j] = MAX3(diag[i-2][j-1]+isMatch, diag[i-1][j-1]+GAP, diag[i-1][j]+GAP);
		cu++;
		maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
	}

#else
	for(j=jstart;j<=jstop;j++)
	{
		int isMatch = Q[i-j-1]==D[diagDoffset+j-1]?MATCH:MISMATCH;

		diag[i][j] = MAX3(diag[i-2][j-1]+isMatch, diag[i-1][j-1]+GAP, diag[i-1][j]+GAP);
		cu++;
		maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
	}
#endif

	threadData->antidiagonalData.cu += cu;
	threadData->antidiagonalData.maxVal = threadData->antidiagonalData.maxVal>=maxVal?threadData->antidiagonalData.maxVal:maxVal;
}

static inline void computeAntidiagonal_P2_A (threadData_t * threadData)
{
	int threadID = threadData->threadID;
	int totalThreads = threadData->threadTOTAL;

	int diagSize = threadData->antidiagonalData.diagSize;
	int diagDoffset = threadData->antidiagonalData.diagDoffset;
	int MATCH = threadData->antidiagonalData.MATCH;
	int MISMATCH = threadData->antidiagonalData.MISMATCH;
	int GAP = threadData->antidiagonalData.GAP;
	char * Q = threadData->antidiagonalData.Q;
	char * D = threadData->antidiagonalData.D;
	int ** diag = threadData->antidiagonalData.diag;
	int i = threadData->antidiagonalData.diagIndex;
	int jstart0 = threadData->antidiagonalData.jstart0;
	int Qindex = threadData->antidiagonalData.Qindex;
	int cu = 0;
	int maxVal = 0;
#ifdef JAM
	int maxVal1 = 0, maxVal2 = 0;
#endif

	int tasksPerThread = diagSize / totalThreads;

	int jstart = tasksPerThread*threadID;
	int jstop = tasksPerThread*threadID+tasksPerThread-1;

	if(threadID==0)
		jstart = jstart0;

	if(threadID==totalThreads-1)
		jstop = diagSize-1-1;
	
	int j;
#ifdef UNROLL2
	int unroll=2;
#ifndef JAM
	for(j=jstart;j<jstop-unroll;j=j+unroll)
	{
		int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;

		diag[i][j] = MAX3(diag[i-2][j]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
		cu++;
		maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;

		isMatch = Q[Qindex-(j+1)]==D[diagDoffset+(j+1)-1]?MATCH:MISMATCH;

		diag[i][(j+1)] = MAX3(diag[i-2][(j+1)]+isMatch, diag[i-1][(j+1)]+GAP, diag[i-1][(j+1)+1]+GAP);
		cu++;
		maxVal = diag[i][(j+1)]>=maxVal?diag[i][(j+1)]:maxVal;
	}
#else
	for(j=jstart;j<jstop-unroll;j=j+unroll)
	{
		int isMatch1 = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;
		int isMatch2 = Q[Qindex-(j+1)]==D[diagDoffset+(j+1)-1]?MATCH:MISMATCH;


		diag[i][j] = MAX3(diag[i-2][j]+isMatch1, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
		diag[i][(j+1)] = MAX3(diag[i-2][(j+1)]+isMatch2, diag[i-1][(j+1)]+GAP, diag[i-1][(j+1)+1]+GAP);

		cu=cu+2;
		maxVal1 = diag[i][j]>=maxVal1?diag[i][j]:maxVal1;
		maxVal2 = diag[i][(j+1)]>=maxVal2?diag[i][(j+1)]:maxVal2;
	}
	maxVal = maxVal1>=maxVal2?maxVal1:maxVal2;
#endif
	for(;j<=jstop;j++)
	{
		int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;

		diag[i][j] = MAX3(diag[i-2][j]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
		cu++;
		maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
	}
#else
	for(j=jstart;j<=jstop;j++)
	{
		int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;

		diag[i][j] = MAX3(diag[i-2][j]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
		cu++;
		maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
	}
#endif
	threadData->antidiagonalData.cu += cu;
	threadData->antidiagonalData.maxVal = threadData->antidiagonalData.maxVal>=maxVal?threadData->antidiagonalData.maxVal:maxVal;
}

static inline void computeAntidiagonal_P2_B (threadData_t * threadData)
{
	int threadID = threadData->threadID;
	int totalThreads = threadData->threadTOTAL;

	int diagSize = threadData->antidiagonalData.diagSize;
	int diagDoffset = threadData->antidiagonalData.diagDoffset;
	int MATCH = threadData->antidiagonalData.MATCH;
	int MISMATCH = threadData->antidiagonalData.MISMATCH;
	int GAP = threadData->antidiagonalData.GAP;
	char * Q = threadData->antidiagonalData.Q;
	char * D = threadData->antidiagonalData.D;
	int ** diag = threadData->antidiagonalData.diag;
	int i = threadData->antidiagonalData.diagIndex;
	int jstart0 = threadData->antidiagonalData.jstart0;
	int Qindex = threadData->antidiagonalData.Qindex;
	int cu = 0;
	int maxVal = 0;
#ifdef JAM
	int maxVal1 = 0, maxVal2 = 0;
#endif

	int tasksPerThread = diagSize / totalThreads;

	int jstart = tasksPerThread*threadID;
	int jstop = tasksPerThread*threadID+tasksPerThread-1;

	if(threadID==0)
		jstart = jstart0;

	if(threadID==totalThreads-1)
		jstop = diagSize-1-1;
	
	int j;
#ifdef UNROLL2
	int unroll = 2;
#ifndef JAM
	for(j=jstart;j<jstop-unroll;j=j+unroll)
	{
		int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;

		diag[i][j] = MAX3(diag[i-2][j+1]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
		cu++;
		maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;

		isMatch = Q[Qindex-(j+1)]==D[diagDoffset+(j+1)-1]?MATCH:MISMATCH;

		diag[i][(j+1)] = MAX3(diag[i-2][(j+1)+1]+isMatch, diag[i-1][(j+1)]+GAP, diag[i-1][(j+1)+1]+GAP);
		cu++;
		maxVal = diag[i][(j+1)]>=maxVal?diag[i][(j+1)]:maxVal;
	}
#else
	for(j=jstart;j<jstop-unroll;j=j+unroll)
	{
		int isMatch1 = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;
		int isMatch2 = Q[Qindex-(j+1)]==D[diagDoffset+(j+1)-1]?MATCH:MISMATCH;

		diag[i][j] = MAX3(diag[i-2][j+1]+isMatch1, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
		diag[i][(j+1)] = MAX3(diag[i-2][(j+1)+1]+isMatch2, diag[i-1][(j+1)]+GAP, diag[i-1][(j+1)+1]+GAP);

		cu=cu+2;
		maxVal1 = diag[i][j]>=maxVal1?diag[i][j]:maxVal1;
		maxVal2 = diag[i][(j+1)]>=maxVal2?diag[i][(j+1)]:maxVal2;
	}
	maxVal = maxVal1>=maxVal2?maxVal1:maxVal2;
#endif
	for(;j<=jstop;j++)
	{
		int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;

		diag[i][j] = MAX3(diag[i-2][j+1]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
		cu++;
		maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
	}

#else
	for(j=jstart;j<=jstop;j++)
	{
		int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;

		diag[i][j] = MAX3(diag[i-2][j+1]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
		cu++;
		maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
	}
#endif

	threadData->antidiagonalData.cu += cu;
	threadData->antidiagonalData.maxVal = threadData->antidiagonalData.maxVal>=maxVal?threadData->antidiagonalData.maxVal:maxVal;
}

static inline void computeAntidiagonal_P3 (threadData_t * threadData)
{
	int threadID = threadData->threadID;
	int totalThreads = threadData->threadTOTAL;

	int diagSize = threadData->antidiagonalData.diagSize;
	int diagDoffset = threadData->antidiagonalData.diagDoffset;
	int MATCH = threadData->antidiagonalData.MATCH;
	int MISMATCH = threadData->antidiagonalData.MISMATCH;
	int GAP = threadData->antidiagonalData.GAP;
	char * Q = threadData->antidiagonalData.Q;
	char * D = threadData->antidiagonalData.D;
	int ** diag = threadData->antidiagonalData.diag;
	int i = threadData->antidiagonalData.diagIndex;
	int jstart0 = threadData->antidiagonalData.jstart0;
	int Qindex = threadData->antidiagonalData.Qindex;
	int cu = 0;
	int maxVal = 0;
#ifdef JAM
	int maxVal1 = 0, maxVal2 = 0;
#endif
	int tasksPerThread = diagSize / totalThreads;

	int jstart = tasksPerThread*threadID;
	int jstop = tasksPerThread*threadID+tasksPerThread-1;

	if(threadID==0)
		jstart = jstart0;

	if(threadID==totalThreads-1)
		jstop = diagSize-1;
	
	int j;
#ifdef UNROLL2
	int unroll = 2;
#ifndef JAM
	for(j=jstart;j<jstop-unroll;j=j+unroll)
	{
		int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;

		diag[i][j] = MAX3(diag[i-2][j+1]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
		cu++;
		maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;

		isMatch = Q[Qindex-(j+1)]==D[diagDoffset+(j+1)-1]?MATCH:MISMATCH;

		diag[i][(j+1)] = MAX3(diag[i-2][(j+1)+1]+isMatch, diag[i-1][(j+1)]+GAP, diag[i-1][(j+1)+1]+GAP);
		cu++;
		maxVal = diag[i][(j+1)]>=maxVal?diag[i][(j+1)]:maxVal;
	}
#else
	for(j=jstart;j<jstop-unroll;j=j+unroll)
	{
		int isMatch1 = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;
		int isMatch2 = Q[Qindex-(j+1)]==D[diagDoffset+(j+1)-1]?MATCH:MISMATCH;

		diag[i][j] = MAX3(diag[i-2][j+1]+isMatch1, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
		diag[i][(j+1)] = MAX3(diag[i-2][(j+1)+1]+isMatch2, diag[i-1][(j+1)]+GAP, diag[i-1][(j+1)+1]+GAP);

		cu=cu+2;
		maxVal1 = diag[i][j]>=maxVal1?diag[i][j]:maxVal1;
		maxVal2 = diag[i][(j+1)]>=maxVal2?diag[i][(j+1)]:maxVal2;
	}
	maxVal = maxVal1>=maxVal2?maxVal1:maxVal2;
#endif
	for(;j<=jstop;j++)
	{
		int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;

		diag[i][j] = MAX3(diag[i-2][j+1]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
		cu++;
		maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
	}
#else
	for(j=jstart;j<=jstop;j++)
	{
		int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;

		diag[i][j] = MAX3(diag[i-2][j+1]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
		cu++;
		maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
	}
#endif
	threadData->antidiagonalData.cu += cu;
	threadData->antidiagonalData.maxVal = threadData->antidiagonalData.maxVal>=maxVal?threadData->antidiagonalData.maxVal:maxVal;
}


static inline void execFunctionMaster(threadData_t * threadData, int operation)
{
	if(operation==COMPUTE_ANTIDIAGONAL)
		computeAntidiagonal(&threadData[0]);

	if(operation==COMPUTE_ANTIDIAGONAL_P2_A)
		computeAntidiagonal_P2_A(&threadData[0]);

	if(operation==COMPUTE_ANTIDIAGONAL_P2_B)
		computeAntidiagonal_P2_B(&threadData[0]);

	if(operation==COMPUTE_ANTIDIAGONAL_P3)
		computeAntidiagonal_P3(&threadData[0]);

}

static inline void setThreadOperation(threadData_t * threadData, int operation)
{
	int i, threads=threadData[0].threadTOTAL;
	
	for(i=0;i<threads;i++)
		threadData[i].threadOPERATION = operation;	
}

void startThreadOperations(threadData_t * threadData, int operation)
{
	setThreadOperation(threadData, operation);

	execFunctionMaster(threadData,operation);

	threadData[0].threadBARRIER=1;
	syncThreadsBARRIER(threadData);		
}

#ifdef PINNING
static void pinToCore(int tid)
{
	cpu_set_t cpuset;
         
	CPU_ZERO(&cpuset);    
	CPU_SET(tid, &cpuset);
	if(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0)
	{
		fprintf(stdout, "\n ERROR: Please specify a number of threads that is smaller or equal");
		fprintf(stdout, "\n        to the number of available physical cores (%d).\n\n",tid);
		exit(0);
	}
}
#endif

void * thread (void * x)
{
	threadData_t * currentThread = (threadData_t *) x;

	int tid = currentThread->threadID;

#ifdef PINNING
	pinToCore(tid);
#endif
	int threads = currentThread->threadTOTAL;
	
	while(1)
	{
		__sync_synchronize();

		if(currentThread->threadOPERATION==EXIT)
			return NULL;
		
		if(currentThread->threadOPERATION==COMPUTE_ANTIDIAGONAL)
		{
			computeAntidiagonal (currentThread);

			currentThread->threadOPERATION=BUSYWAIT;

#ifdef SENSE_REVERSAL_BARRIER
			sense_reversal_barrier(tid, threads);
#else
#ifdef PTHREAD_BARRIER
 			pthread_barrier_wait(&barrier);
#else
			currentThread->threadBARRIER=1;			
			while(currentThread->threadBARRIER==1) __sync_synchronize();
#endif
#endif
		}

		if(currentThread->threadOPERATION==COMPUTE_ANTIDIAGONAL_P2_A)
		{
			computeAntidiagonal_P2_A (currentThread);

			currentThread->threadOPERATION=BUSYWAIT;

#ifdef SENSE_REVERSAL_BARRIER
			sense_reversal_barrier(tid, threads);
#else
#ifdef PTHREAD_BARRIER
 			pthread_barrier_wait(&barrier);
#else
			currentThread->threadBARRIER=1;			
			while(currentThread->threadBARRIER==1) __sync_synchronize();
#endif
#endif		
		}

		if(currentThread->threadOPERATION==COMPUTE_ANTIDIAGONAL_P2_B)
		{
			computeAntidiagonal_P2_B (currentThread);

			currentThread->threadOPERATION=BUSYWAIT;

#ifdef SENSE_REVERSAL_BARRIER
			sense_reversal_barrier(tid, threads);
#else
#ifdef PTHREAD_BARRIER
 			pthread_barrier_wait(&barrier);
#else
			currentThread->threadBARRIER=1;			
			while(currentThread->threadBARRIER==1) __sync_synchronize();
#endif
#endif		
		}

		if(currentThread->threadOPERATION==COMPUTE_ANTIDIAGONAL_P3)
		{
			computeAntidiagonal_P3 (currentThread);

			currentThread->threadOPERATION=BUSYWAIT;

#ifdef SENSE_REVERSAL_BARRIER
			sense_reversal_barrier(tid, threads);
#else
#ifdef PTHREAD_BARRIER
 			pthread_barrier_wait(&barrier);
#else
			currentThread->threadBARRIER=1;			
			while(currentThread->threadBARRIER==1) __sync_synchronize();
#endif
#endif		
		}

	}
	
	return NULL;		
}

void terminateWorkerThreads(pthread_t * workerThreadL, threadData_t * threadData)
{
	int i, threads=threadData[0].threadTOTAL;
	
	for(i=0;i<threads;i++)
		threadData[i].threadOPERATION = EXIT;			

	for(i=1;i<threads;i++)
		pthread_join(workerThreadL[i-1],NULL);
}

#endif




int main()
{	
	char alphabet[4]="abcd";

	srand(1);

	int i=0, j=0, iterIndex=0, itersTotal = ITERATIONS;

	int MATCH = 3, MISMATCH = -1, GAP = -1;

	int sizeD = DSIZE;
	char * D = (char*)malloc(sizeof(char)*(unsigned long)sizeD);
	assert(D!=NULL);

	for(i=0;i<sizeD;i++)
		D[i]=alphabet[rand()%4];

	int sizeQ = QSIZE;
	char * Q = (char*)malloc(sizeof(char)*(unsigned long)sizeQ);
	assert(Q!=NULL);

	assert(sizeQ<=sizeD);

	for(i=0;i<sizeQ;i++)
		Q[i]=alphabet[rand()%4];

	int rowSize = sizeD+1;
	int colSize = sizeQ+1;

	int * matMem = (int *)malloc(sizeof(int)*(unsigned long)rowSize*(unsigned long)colSize);
	assert(matMem!=NULL);

	for(i=0;i<rowSize*colSize;i++)
		matMem[i] = 0;

	int cu = 0;

#if defined(REFERENCE) || defined(ANTIDIAGONALS)
	int ** mat = (int**)malloc(sizeof(int*)*(unsigned long)colSize);
	assert(mat!=NULL);

	for(i=0;i<colSize;i++)
	{
		mat[i] = &(matMem[i*rowSize]); 
	}
#endif	
#ifdef ANTIDIAGONALS_WITH_MLT
	//MLT
	int diagonalsTotal = colSize+rowSize-1;
	int ** diag = (int**)malloc(sizeof(int*)*(unsigned long)diagonalsTotal);
	assert(diag!=NULL);

	int diagsP1 = colSize - 1;
	int diagsP2 = rowSize - colSize + 1;
	int diagsP3 = diagsP1;

	int curDiagOffset = 0;

	// Part 1
	diag[0] = &(matMem[curDiagOffset]);
	for(i=1;i<diagsP1;i++)
	{
		int diagSize = i; 
		curDiagOffset += diagSize;
		diag[i] = &(matMem[curDiagOffset]);
	} 

	// Part 2
	int diagSizeFxd = colSize;
	for(i=diagsP1;i<diagsP1+diagsP2;i++)
	{
		curDiagOffset += diagSizeFxd;
		diag[i] = &(matMem[curDiagOffset]);
	}

	// Part 3
	for(i=diagsP1+diagsP2;i<diagsP1+diagsP2+diagsP3;i++)
	{
		int diagSize = diagSizeFxd--;
		curDiagOffset += diagSize;		
		diag[i] = &(matMem[curDiagOffset]);
	}
#endif

#ifdef PTHREADS

	int threads = THREADS;

#ifdef SENSE_REVERSAL_BARRIER
	sense_reversal_barrier_init (threads);
#endif

#ifdef PTHREAD_BARRIER
	int s = pthread_barrier_init(&barrier, NULL, (unsigned int)threads);
	assert(s == 0);
#endif
	assert(sizeQ>threads);

	workerThread = NULL; 
	workerThread = (pthread_t *) malloc (sizeof(pthread_t)*((unsigned long)(threads-1)));
	
	threadData_t * threadData = (threadData_t *) malloc (sizeof(threadData_t)*((unsigned long)threads));
	assert(threadData!=NULL);

	for(i=0;i<threads;i++)
		initializeThreadData(&threadData[i],i,threads, MATCH, MISMATCH, GAP, Q, D, diag, sizeQ);

	for(i=1;i<threads;i++)
		pthread_create (&workerThread[i-1], NULL, thread, (void *) (&threadData[i]));


	

#endif

	double time0 = gettime();
	int maxVal = 0;

	for(iterIndex=0;iterIndex<itersTotal;iterIndex++)
	{
		maxVal=0;
		cu = 0;

#ifdef JAM
		int maxVal1 = 0, maxVal2 = 0, maxVal3 = 0, maxVal4 = 0;
#endif

#ifdef PTHREADS
	for(i=0;i<threads;i++)
		initializeThreadData(&threadData[i],i,threads, MATCH, MISMATCH, GAP, Q, D, diag, sizeQ);

#endif

#ifdef REFERENCE

		for(i=1;i<colSize;i++)
		{
#ifdef UNROLL4
			int unroll = 4;
#ifndef JAM
			
			for(j=1;j<rowSize-unroll;j=j+unroll)
			{	
				int isMatch = Q[i-1]==D[j-1]?MATCH:MISMATCH;

				mat[i][j] = MAX3(mat[i-1][j-1]+isMatch, mat[i][j-1]+GAP, mat[i-1][j]+GAP);
				cu++;
				maxVal = mat[i][j]>=maxVal?mat[i][j]:maxVal;
				
				isMatch = Q[i-1]==D[(j+1)-1]?MATCH:MISMATCH;

				mat[i][(j+1)] = MAX3(mat[i-1][(j+1)-1]+isMatch, mat[i][(j+1)-1]+GAP, mat[i-1][(j+1)]+GAP);
				cu++;
				maxVal = mat[i][(j+1)]>=maxVal?mat[i][(j+1)]:maxVal;

				isMatch = Q[i-1]==D[(j+2)-1]?MATCH:MISMATCH;

				mat[i][(j+2)] = MAX3(mat[i-1][(j+2)-1]+isMatch, mat[i][(j+2)-1]+GAP, mat[i-1][(j+2)]+GAP);
				cu++;
				maxVal = mat[i][(j+2)]>=maxVal?mat[i][(j+2)]:maxVal;

				isMatch = Q[i-1]==D[(j+3)-1]?MATCH:MISMATCH;

				mat[i][(j+3)] = MAX3(mat[i-1][(j+3)-1]+isMatch, mat[i][(j+3)-1]+GAP, mat[i-1][(j+3)]+GAP);
				cu++;
				maxVal = mat[i][(j+3)]>=maxVal?mat[i][(j+3)]:maxVal;
			}
#else
			for(j=1;j<rowSize-unroll;j=j+unroll)
			{	
				int isMatch1 = Q[i-1]==D[j-1]?MATCH:MISMATCH;
				int isMatch2 = Q[i-1]==D[j]?MATCH:MISMATCH;
				int isMatch3 = Q[i-1]==D[j+1]?MATCH:MISMATCH;
				int isMatch4 = Q[i-1]==D[j+2]?MATCH:MISMATCH;

				mat[i][j] = MAX3(mat[i-1][j-1]+isMatch1, mat[i][j-1]+GAP, mat[i-1][j]+GAP);
				mat[i][j+1] = MAX3(mat[i-1][j]+isMatch2, mat[i][j]+GAP, mat[i-1][j+1]+GAP);
				mat[i][j+2] = MAX3(mat[i-1][j+1]+isMatch3, mat[i][j+1]+GAP, mat[i-1][j+2]+GAP);
				mat[i][j+3] = MAX3(mat[i-1][j+2]+isMatch4, mat[i][j+2]+GAP, mat[i-1][j+3]+GAP);

				cu=cu+4;
				maxVal1 = mat[i][j]>=maxVal1?mat[i][j]:maxVal1;
				maxVal2 = mat[i][j+1]>=maxVal2?mat[i][j+1]:maxVal2;
				maxVal3 = mat[i][j+2]>=maxVal3?mat[i][j+2]:maxVal3;
				maxVal4 = mat[i][j+3]>=maxVal4?mat[i][j+3]:maxVal4;
			}
			maxVal1 = maxVal1>=maxVal2?maxVal1:maxVal2;
			maxVal3 = maxVal3>=maxVal4?maxVal3:maxVal4;
			maxVal = maxVal1>=maxVal3?maxVal1:maxVal3;

#endif
			for(;j<rowSize;j++)
			{	
				int isMatch = Q[i-1]==D[j-1]?MATCH:MISMATCH;

				mat[i][j] = MAX3(mat[i-1][j-1]+isMatch, mat[i][j-1]+GAP, mat[i-1][j]+GAP);
				cu++;
				maxVal = mat[i][j]>=maxVal?mat[i][j]:maxVal;
			}
#else
			for(j=1;j<rowSize;j++)
			{	
				int isMatch = Q[i-1]==D[j-1]?MATCH:MISMATCH;

				mat[i][j] = MAX3(mat[i-1][j-1]+isMatch, mat[i][j-1]+GAP, mat[i-1][j]+GAP);
				cu++;
				maxVal = mat[i][j]>=maxVal?mat[i][j]:maxVal;
			}
#endif
		}
#endif
#ifdef ANTIDIAGONALS
		// Part 1
		int diagsP1 = sizeQ - 1;
		int dj_start = 1;
		for(i=0;i<diagsP1;i++)
		{
			int diagSize = i+1;
			int di_start = i+1;
	 
			for(j=0;j<diagSize;j++)
			{
				int ti = di_start-j;
				int tj = dj_start+j;

				int isMatch = Q[ti-1]==D[tj-1]?MATCH:MISMATCH;

				mat[ti][tj] = MAX3(mat[ti-1][tj-1]+isMatch, mat[ti][tj-1]+GAP, mat[ti-1][tj]+GAP);
				cu++;
				maxVal = mat[ti][tj]>=maxVal?mat[ti][tj]:maxVal;
			}
		}
 
		// Part 2
		int diagsP2 = sizeD - sizeQ + 1;
		dj_start = 1;
		int di_start = sizeQ;
		int diagSizeFxd = sizeQ;
		for(i=0;i<diagsP2;i++)
		{
			dj_start = i + 1;

			for(j=0;j<diagSizeFxd;j++)
			{
				int ti = di_start-j;
				int tj = dj_start+j;

				int isMatch = Q[ti-1]==D[tj-1]?MATCH:MISMATCH;

				mat[ti][tj] = MAX3(mat[ti-1][tj-1]+isMatch, mat[ti][tj-1]+GAP, mat[ti-1][tj]+GAP);
				cu++;
				maxVal = mat[ti][tj]>=maxVal?mat[ti][tj]:maxVal;
			}
		}

		// Part 3
		int diagsP3 = diagsP1;
		di_start = sizeQ;
		for(i=0;i<diagsP3;i++)
		{
			int diagSize = sizeQ-i-1;
			int dj_start = sizeD-sizeQ+2+i;
	 
			for(j=0;j<diagSize;j++)
			{
				int ti = di_start-j;
				int tj = dj_start+j;
			
				int isMatch = Q[ti-1]==D[tj-1]?MATCH:MISMATCH;

				mat[ti][tj] = MAX3(mat[ti-1][tj-1]+isMatch, mat[ti][tj-1]+GAP, mat[ti-1][tj]+GAP);
				cu++;
				maxVal = mat[ti][tj]>=maxVal?mat[ti][tj]:maxVal;
			}
		}
#endif
#ifdef ANTIDIAGONALS_WITH_MLT

#ifndef PTHREADS
		// Part 1 with MLT
		int diagDoffset = 0;
		for(i=0;i<=diagsP1;i++) 
		{	
			int diagSize = i+1;
#ifdef UNROLL4
			int unroll = 4;
#ifndef JAM
			for(j=1;j<diagSize-1-unroll;j=j+unroll)
			{
				int isMatch = Q[i-j-1]==D[diagDoffset+j-1]?MATCH:MISMATCH;

				diag[i][j] = MAX3(diag[i-2][j-1]+isMatch, diag[i-1][j-1]+GAP, diag[i-1][j]+GAP);
				cu++;
				maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;

				isMatch = Q[i-(j+1)-1]==D[diagDoffset+(j+1)-1]?MATCH:MISMATCH;

				diag[i][(j+1)] = MAX3(diag[i-2][(j+1)-1]+isMatch, diag[i-1][(j+1)-1]+GAP, diag[i-1][(j+1)]+GAP);
				cu++;
				maxVal = diag[i][(j+1)]>=maxVal?diag[i][(j+1)]:maxVal;

				
				isMatch = Q[i-(j+2)-1]==D[diagDoffset+(j+2)-1]?MATCH:MISMATCH;

				diag[i][(j+2)] = MAX3(diag[i-2][(j+2)-1]+isMatch, diag[i-1][(j+2)-1]+GAP, diag[i-1][(j+2)]+GAP);
				cu++;
				maxVal = diag[i][(j+2)]>=maxVal?diag[i][(j+2)]:maxVal;

				isMatch = Q[i-(j+3)-1]==D[diagDoffset+(j+3)-1]?MATCH:MISMATCH;

				diag[i][(j+3)] = MAX3(diag[i-2][(j+3)-1]+isMatch, diag[i-1][(j+3)-1]+GAP, diag[i-1][(j+3)]+GAP);
				cu++;
				maxVal = diag[i][(j+3)]>=maxVal?diag[i][(j+3)]:maxVal;
			}
#else
			for(j=1;j<diagSize-1-unroll;j=j+unroll)
			{
				int isMatch1 = Q[i-j-1]==D[diagDoffset+j-1]?MATCH:MISMATCH;
				int isMatch2 = Q[i-(j+1)-1]==D[diagDoffset+(j+1)-1]?MATCH:MISMATCH;
				int isMatch3 = Q[i-(j+2)-1]==D[diagDoffset+(j+2)-1]?MATCH:MISMATCH;
				int isMatch4 = Q[i-(j+3)-1]==D[diagDoffset+(j+3)-1]?MATCH:MISMATCH;

				diag[i][j] = MAX3(diag[i-2][j-1]+isMatch1, diag[i-1][j-1]+GAP, diag[i-1][j]+GAP);
				diag[i][(j+1)] = MAX3(diag[i-2][(j+1)-1]+isMatch2, diag[i-1][(j+1)-1]+GAP, diag[i-1][(j+1)]+GAP);
				diag[i][(j+2)] = MAX3(diag[i-2][(j+2)-1]+isMatch3, diag[i-1][(j+2)-1]+GAP, diag[i-1][(j+2)]+GAP);
				diag[i][(j+3)] = MAX3(diag[i-2][(j+3)-1]+isMatch4, diag[i-1][(j+3)-1]+GAP, diag[i-1][(j+3)]+GAP);

				cu=cu+4;
				maxVal1 = diag[i][j]>=maxVal1?diag[i][j]:maxVal1;			
				maxVal2 = diag[i][(j+1)]>=maxVal2?diag[i][(j+1)]:maxVal2;			
				maxVal3 = diag[i][(j+2)]>=maxVal3?diag[i][(j+2)]:maxVal3;			
				maxVal4 = diag[i][(j+3)]>=maxVal4?diag[i][(j+3)]:maxVal4;
			}
			//maxVal1 = maxVal1>=maxVal2?maxVal1:maxVal2;
			//maxVal3 = maxVal3>=maxVal4?maxVal3:maxVal4;
			//maxVal = maxVal1>=maxVal3?maxVal1:maxVal3;
#endif
			for(;j<diagSize-1;j++)
			{
				int isMatch = Q[i-j-1]==D[diagDoffset+j-1]?MATCH:MISMATCH;

				diag[i][j] = MAX3(diag[i-2][j-1]+isMatch, diag[i-1][j-1]+GAP, diag[i-1][j]+GAP);
				cu++;
				maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
			}
#else
			for(j=1;j<diagSize-1;j++)
			{
				int isMatch = Q[i-j-1]==D[diagDoffset+j-1]?MATCH:MISMATCH;

				diag[i][j] = MAX3(diag[i-2][j-1]+isMatch, diag[i-1][j-1]+GAP, diag[i-1][j]+GAP);
				cu++;
				maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
			}

#endif
		}
#else		// PTHREADS

		// Part 1 with MLT and PTHREADS
		int diagDoffset = 0;

		for(i=0;i<threads-1;i++) 
		{	
			int diagSize = i+1;

			for(j=1;j<diagSize-1;j++)
			{
				int isMatch = Q[i-j-1]==D[diagDoffset+j-1]?MATCH:MISMATCH;

				diag[i][j] = MAX3(diag[i-2][j-1]+isMatch, diag[i-1][j-1]+GAP, diag[i-1][j]+GAP);
				cu++;
				maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
			}
		}

		for(i=threads-1;i<=diagsP1;i++) 
		{	
			int diagSize = i+1;
		
			updateThreadArguments_COMPUTE_ANTIDIAGONAL_operation(threadData, diagSize, diagDoffset, i, 1);
			startThreadOperations(threadData, COMPUTE_ANTIDIAGONAL);	
		}
#endif

#ifdef ANTIDIAGONALS_WITH_MLT_NO_IF
#ifndef PTHREADS
		// Part 2 with MLT
		i=diagsP1+1;
		{	
			diagDoffset++;
			int diagSize = colSize;
			int Qindex = sizeQ-1;
#ifdef UNROLL4
			int unroll = 4;
#ifndef JAM
			for(j=0;j<diagSize-1-unroll;j=j+unroll)
			{
				int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;
	
				diag[i][j] = MAX3(diag[i-2][j]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
		
				cu++;
				maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;

				isMatch = Q[Qindex-(j+1)]==D[diagDoffset+(j+1)-1]?MATCH:MISMATCH;
	
				diag[i][(j+1)] = MAX3(diag[i-2][(j+1)]+isMatch, diag[i-1][(j+1)]+GAP, diag[i-1][(j+1)+1]+GAP);
		
				cu++;
				maxVal = diag[i][(j+1)]>=maxVal?diag[i][(j+1)]:maxVal;

				isMatch = Q[Qindex-(j+2)]==D[diagDoffset+(j+2)-1]?MATCH:MISMATCH;
	
				diag[i][(j+2)] = MAX3(diag[i-2][(j+2)]+isMatch, diag[i-1][(j+2)]+GAP, diag[i-1][(j+2)+1]+GAP);
		
				cu++;
				maxVal = diag[i][(j+2)]>=maxVal?diag[i][(j+2)]:maxVal;

				isMatch = Q[Qindex-(j+3)]==D[diagDoffset+(j+3)-1]?MATCH:MISMATCH;
	
				diag[i][(j+3)] = MAX3(diag[i-2][(j+3)]+isMatch, diag[i-1][(j+3)]+GAP, diag[i-1][(j+3)+1]+GAP);
		
				cu++;
				maxVal = diag[i][(j+3)]>=maxVal?diag[i][(j+3)]:maxVal;
			}
#else
			for(j=0;j<diagSize-1-unroll;j=j+unroll)
			{
				int isMatch1 = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;
				int isMatch2 = Q[Qindex-(j+1)]==D[diagDoffset+(j+1)-1]?MATCH:MISMATCH;
				int isMatch3 = Q[Qindex-(j+2)]==D[diagDoffset+(j+2)-1]?MATCH:MISMATCH;
				int isMatch4 = Q[Qindex-(j+3)]==D[diagDoffset+(j+3)-1]?MATCH:MISMATCH;


				diag[i][j] = MAX3(diag[i-2][j]+isMatch1, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
				diag[i][(j+1)] = MAX3(diag[i-2][(j+1)]+isMatch2, diag[i-1][(j+1)]+GAP, diag[i-1][(j+1)+1]+GAP);
				diag[i][(j+2)] = MAX3(diag[i-2][(j+2)]+isMatch3, diag[i-1][(j+2)]+GAP, diag[i-1][(j+2)+1]+GAP);
				diag[i][(j+3)] = MAX3(diag[i-2][(j+3)]+isMatch4, diag[i-1][(j+3)]+GAP, diag[i-1][(j+3)+1]+GAP);
		
				cu=cu+4;
				maxVal1 = diag[i][j]>=maxVal1?diag[i][j]:maxVal1;		
				maxVal2 = diag[i][(j+1)]>=maxVal2?diag[i][(j+1)]:maxVal2;		
				maxVal3 = diag[i][(j+2)]>=maxVal3?diag[i][(j+2)]:maxVal3;		
				maxVal4 = diag[i][(j+3)]>=maxVal4?diag[i][(j+3)]:maxVal4;
			}
			//maxVal1 = maxVal1>=maxVal2?maxVal1:maxVal2;
			//maxVal3 = maxVal3>=maxVal4?maxVal3:maxVal4;
			//maxVal = maxVal1>=maxVal3?maxVal1:maxVal3;
#endif
			for(;j<diagSize-1;j++)
			{
				int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;
	
				diag[i][j] = MAX3(diag[i-2][j]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
		
				cu++;
				maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
			}
#else
			for(j=0;j<diagSize-1;j++)
			{
				int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;
	
				diag[i][j] = MAX3(diag[i-2][j]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
		
				cu++;
				maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
			}
#endif			
		}
		i++;
		for(;i<diagsP1+diagsP2;i++) 
		{	
			diagDoffset++;
			int diagSize = colSize;
			int Qindex = sizeQ-1;
			for(j=0;j<diagSize-1;j++)
			{
				
				int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;
	
				diag[i][j] = MAX3(diag[i-2][j+1]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
				
				cu++;
				maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
			}			
		}
#else
		// Part 2 with MLT
		i=diagsP1+1;
		{	
			diagDoffset++;
			int diagSize = colSize;
			//int Qindex = sizeQ-1;

			updateThreadArguments_COMPUTE_ANTIDIAGONAL_operation(threadData, diagSize, diagDoffset, i, 0);
			startThreadOperations(threadData, COMPUTE_ANTIDIAGONAL_P2_A);						
		}
		i++;
		for(;i<diagsP1+diagsP2;i++) 
		{	
			diagDoffset++;
			int diagSize = colSize;
			//int Qindex = sizeQ-1;

			updateThreadArguments_COMPUTE_ANTIDIAGONAL_operation(threadData, diagSize, diagDoffset, i, 0);
			startThreadOperations(threadData, COMPUTE_ANTIDIAGONAL_P2_B);					
		}
#endif
#else		
		// Part 2 with MLT
		for(i=diagsP1+1;i<diagsP1+diagsP2;i++) 
		{	
			diagDoffset++;
			int diagSize = colSize;
			int Qindex = sizeQ-1;
			for(j=0;j<diagSize-1;j++)
			{
				
				int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;

				if(i==diagsP1+1)
					diag[i][j] = MAX3(diag[i-2][j]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
				else
					diag[i][j] = MAX3(diag[i-2][j+1]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
				
				cu++;
				maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
			}			
		}
#endif
#ifndef PTHREADS
		// Part 3 with MLT
		int diagSizeCur = colSize-1;
		for(i=diagsP1+diagsP2;i<diagsP1+diagsP2+diagsP3;i++) 
		{	
			diagDoffset++;
			int diagSize = diagSizeCur--;
			int Qindex = sizeQ-1;
#ifdef UNROLL4
			int unroll = 4;
#ifndef JAM
			for(j=0;j<diagSize-unroll;j=j+unroll) 
			{
				int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;

				diag[i][j] = MAX3(diag[i-2][j+1]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
				
				cu++;
				maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;

				isMatch = Q[Qindex-(j+1)]==D[diagDoffset+(j+1)-1]?MATCH:MISMATCH;

				diag[i][(j+1)] = MAX3(diag[i-2][(j+1)+1]+isMatch, diag[i-1][(j+1)]+GAP, diag[i-1][(j+1)+1]+GAP);
				
				cu++;
				maxVal = diag[i][(j+1)]>=maxVal?diag[i][(j+1)]:maxVal;

				isMatch = Q[Qindex-(j+2)]==D[diagDoffset+(j+2)-1]?MATCH:MISMATCH;

				diag[i][(j+2)] = MAX3(diag[i-2][(j+2)+1]+isMatch, diag[i-1][(j+2)]+GAP, diag[i-1][(j+2)+1]+GAP);
				
				cu++;
				maxVal = diag[i][(j+2)]>=maxVal?diag[i][(j+2)]:maxVal;

				isMatch = Q[Qindex-(j+3)]==D[diagDoffset+(j+3)-1]?MATCH:MISMATCH;

				diag[i][(j+3)] = MAX3(diag[i-2][(j+3)+1]+isMatch, diag[i-1][(j+3)]+GAP, diag[i-1][(j+3)+1]+GAP);
				
				cu++;
				maxVal = diag[i][(j+3)]>=maxVal?diag[i][(j+3)]:maxVal;
			}
#else
			for(j=0;j<diagSize-unroll;j=j+unroll) 
			{
				int isMatch1 = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;
				int isMatch2 = Q[Qindex-(j+1)]==D[diagDoffset+(j+1)-1]?MATCH:MISMATCH;
				int isMatch3 = Q[Qindex-(j+2)]==D[diagDoffset+(j+2)-1]?MATCH:MISMATCH;
				int isMatch4 = Q[Qindex-(j+3)]==D[diagDoffset+(j+3)-1]?MATCH:MISMATCH;

				diag[i][j] = MAX3(diag[i-2][j+1]+isMatch1, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
				diag[i][(j+1)] = MAX3(diag[i-2][(j+1)+1]+isMatch2, diag[i-1][(j+1)]+GAP, diag[i-1][(j+1)+1]+GAP);
				diag[i][(j+2)] = MAX3(diag[i-2][(j+2)+1]+isMatch3, diag[i-1][(j+2)]+GAP, diag[i-1][(j+2)+1]+GAP);
				diag[i][(j+3)] = MAX3(diag[i-2][(j+3)+1]+isMatch4, diag[i-1][(j+3)]+GAP, diag[i-1][(j+3)+1]+GAP);
				
				cu=cu+4;
				maxVal1 = diag[i][j]>=maxVal1?diag[i][j]:maxVal1;	
				maxVal2 = diag[i][(j+1)]>=maxVal2?diag[i][(j+1)]:maxVal2;			
				maxVal3 = diag[i][(j+2)]>=maxVal3?diag[i][(j+2)]:maxVal3;				
				maxVal4 = diag[i][(j+3)]>=maxVal4?diag[i][(j+3)]:maxVal4;
			}
			maxVal1 = maxVal1>=maxVal2?maxVal1:maxVal2;
			maxVal3 = maxVal3>=maxVal4?maxVal3:maxVal4;
			maxVal1 = maxVal1>=maxVal3?maxVal1:maxVal3;
			maxVal = maxVal>=maxVal1?maxVal:maxVal1;
#endif
			for(;j<diagSize;j++) 
			{
				int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;

				diag[i][j] = MAX3(diag[i-2][j+1]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
				
				cu++;
				maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
			}
#else
			for(j=0;j<diagSize;j++) 
			{
				int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;

				diag[i][j] = MAX3(diag[i-2][j+1]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
				
				cu++;
				maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
			}
#endif			
		}
#else
		int diagSizeCur = colSize-1;
		for(i=diagsP1+diagsP2;i<diagsP1+diagsP2+diagsP3-threads;i++) 
		{	
			diagDoffset++;
			int diagSize = diagSizeCur--;
			//int Qindex = sizeQ-1;

			updateThreadArguments_COMPUTE_ANTIDIAGONAL_operation(threadData, diagSize, diagDoffset, i, 0);
			startThreadOperations(threadData, COMPUTE_ANTIDIAGONAL_P3);
		}

		for(i=diagsP1+diagsP2+diagsP3-threads;i<diagsP1+diagsP2+diagsP3;i++) 
		{	
			diagDoffset++;
			int diagSize = diagSizeCur--;
			int Qindex = sizeQ-1;

			for(j=0;j<diagSize;j++) 
			{
				int isMatch = Q[Qindex-j]==D[diagDoffset+j-1]?MATCH:MISMATCH;

				diag[i][j] = MAX3(diag[i-2][j+1]+isMatch, diag[i-1][j]+GAP, diag[i-1][j+1]+GAP);
				
				cu++;
				maxVal = diag[i][j]>=maxVal?diag[i][j]:maxVal;
			}			
		}
#endif	
	
#endif
#ifdef PTHREADS
	for(i=0;i<threads;i++)
	{
		cu += threadData[i].antidiagonalData.cu;
		maxVal = threadData[i].antidiagonalData.maxVal>maxVal?threadData[i].antidiagonalData.maxVal:maxVal;//?threadData->antidiagonalData.maxVal:maxVal;
	}
#endif

#ifdef PRINT_MATRIX
	printf("Cell Updates\t%d\n", cu);
	printf("Max Value\t%d\n", maxVal);
#if defined(REFERENCE)||defined(ANTIDIAGONALS)
	for(i=0;i<colSize;i++)
	{
		if(i!=0)
		{
			printf("%c:\t", Q[i-1]);

			for(j=0;j<rowSize;j++)
			{	
				printf("%d\t", mat[i][j]);
			}
		}
		else
		{
			printf("\t\t");

			for(j=0;j<sizeD;j++)
			{	
				printf("%c\t", D[j]);
			}
			printf("\n\t");
			for(j=0;j<rowSize;j++)
			{	
				printf("%d\t", mat[i][j]);
			}
		}
		
		printf("\n");
	}
#endif
#ifdef ANTIDIAGONALS_WITH_MLT
	// Part 1
	/*for(i=0;i<diagsP1;i++)
	{
		int j, diagSize = i+1;
		printf("Diag1 %d:\t", i);
		for(j=0;j<diagSize;j++)
			 printf("%d\t", diag[i][j]);

		printf("\n");	
	} 

	// Part 2
	diagSizeFxd = colSize;
	for(i=diagsP1;i<diagsP1+diagsP2;i++)
	{
		int diagSize = diagSizeFxd;
		printf("Diag2 %d:\t", i);
		for(j=0;j<diagSize;j++)
			 printf("%d\t", diag[i][j]);

		printf("\n");	
	}

	// Part 3
	int diagSizeCur = colSize-1;
	for(i=diagsP1+diagsP2;i<diagsP1+diagsP2+diagsP3;i++)
	{
		int diagSize = diagSizeCur--;
		printf("Diag3 %d:\t", i);
		for(j=0;j<diagSize;j++)
			 printf("%d\t", diag[i][j]);

		printf("\n");
	}
	*/

	int firstDiagIndex = 0;
	int curDiagSize = 0;
	for(i=0;i<colSize;i++)
	{
		if(i!=0)
		{			
			firstDiagIndex++;
			printf("%c:\t", Q[i-1]);
	
			for(j=0;j<rowSize-i;j++)
			{
				if(firstDiagIndex+j<colSize)
					curDiagSize = firstDiagIndex+j+1;

				printf("%d\t", diag[firstDiagIndex+j][curDiagSize-1-i]);
			}

			int jj = i;
			for(j=rowSize-i;j<rowSize;j++)
			{	jj--;

				curDiagSize--;
				printf("%d\t", diag[firstDiagIndex+j][curDiagSize-1-jj]);
			}			
		}
		else
		{
			printf("\t\t");

			for(j=0;j<sizeD;j++)
			{	
				printf("%c\t", D[j]);
			}
			printf("\n\t");
			for(j=0;j<rowSize;j++)
			{	
				curDiagSize++;
				if(j>=colSize)
					curDiagSize=colSize;

				printf("%d\t", diag[firstDiagIndex+j][curDiagSize-1-i]);
			}
		}		
		printf("\n");
	}
#endif
#endif



	}
	
	double time1 = gettime();
	printf("Elapsed Time\t%f\n", time1-time0);


#ifdef PTHREADS
	terminateWorkerThreads(workerThread,threadData);
#ifdef PTHREAD_BARRIER
	pthread_barrier_destroy(&barrier);
#else
#ifdef SENSE_REVERSAL_BARRIER
	sense_reversal_barrier_destroy();
#else
	if(threadData!=NULL)
		free(threadData);

	threadData = NULL;
#endif
#endif
#endif

	
	if(D!=NULL)
	{
		free(D);
		D=NULL;
	}
	if(Q!=NULL)
	{
		free(Q);
		Q=NULL;
	}

	if(matMem!=NULL)
	{
		free(matMem);
		matMem=NULL;
	}


#if defined(REFERENCE) || defined(ANTIDIAGONALS)
	if(mat!=NULL)
	{
		free(mat);
		mat=NULL;
	}
#endif

#ifdef ANTIDIAGONALS_WITH_MLT
	if(diag!=NULL)
	{
		free(diag);
		diag=NULL;
	}
#endif


}
