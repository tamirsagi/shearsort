/*
shearsort
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <math.h>
#include <time.h>       /* time */
#include <math.h>
#include <algorithm> 

const int MASTER_ID = 0;
const int ORDER_DESCENDING = -1;
const int ORDER_ASCENDING = 1;
const int INDICES = 2;

const int ORDER_COL = 0;
const int ORDER_ROW = 1;

const int ROW_INDEX = 0;
const int COL_INDEX = 1;

const int TAG_SLAVES_TO_SLAVES = 0;
const int TAG_SLAVES_TO_MASTER = 100;
const int TAG_MASTERS_TO_MAIN_MASTER = 1000;

MPI_Comm cart_comm;


int* initializeNxNMatrix(int aLen);
void printArray(int* aArray, int aLen);
int oddEven(int aID, int aValue, int aLen, int aSortingOrder, int aOrderType, int aMaster);
int iteration(int aId, int aMyValue, int aLen, int order, int aOrderType, int aMasterId);
int ShearSort(int aId, int aMainMaster, int aElementsPerRow, int aMyValue);
void PrintArrayAsMatrix(int *aArray, int aLen);

int main(int argc, char *argv[])
{
	int numprocs, myid;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int *nums = NULL;
	int *sortedValues;
	MPI_Init(&argc, &argv);
	int value;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	
	double row = sqrt(numprocs);
	if (row - (int)row > 0){
		printf("Error: Number of proccessor is invalid");
		fflush(stdout);
		MPI_Finalize();
		return 0;
	}
	int dim[] = { row, row };
	int period[] = { 1, 0 };
	int reorder = 0;
	double start_time = 0, end_time = 0;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cart_comm);
	if (myid == MASTER_ID)
	{
		start_time = MPI_Wtime();
		printf("\nStart Time %f\n", start_time);
		fflush(stdout);
		sortedValues = (int*) malloc(sizeof(int) * numprocs);
		nums = initializeNxNMatrix(numprocs);
		printf("Array is:\n");
		fflush(stdout);
		PrintArrayAsMatrix(nums, numprocs);
	}
	MPI_Scatter(nums, 1, MPI_INT, &value, 1, MPI_INT, MASTER_ID, MPI_COMM_WORLD);
	value = ShearSort(myid, MASTER_ID,row ,value);
	//master gets each processor's value and save into buffer
	MPI_Gather(&value, 1, MPI_INT, sortedValues, 1, MPI_INT, MASTER_ID, MPI_COMM_WORLD);
	/*printf("after: \nmyId = %d, value = %d\n", myid, value);
	fflush(stdout);*/
	if (myid == MASTER_ID){
		printf("\nSorted Array is:\n");
		fflush(stdout);
		PrintArrayAsMatrix(sortedValues,numprocs);
		end_time = MPI_Wtime();
		printf("\nEnd Time %f\nTotal Time = %f\n", end_time, (end_time - start_time));
		fflush(stdout);
	}
	MPI_Finalize();

}

int* initializeNxNMatrix(int aLen)
{
	static int* randNums = (int*) malloc(sizeof(int)* aLen);
	srand(time(NULL));
	for (int i = 0; i < aLen; i++){
		randNums[i] = rand() % 100;
	}
	return randNums;
}

void printArray(int *aArray, int aLen){
	for (int i = 0; i < aLen; i++){
		printf("%d ", aArray[i]);
		fflush(stdout);
	}
}

void PrintArrayAsMatrix(int *aArray, int aLen){
	int sqr = sqrt(aLen);
	for (int i = 0; i < aLen; i++){
		if (i % sqr == 0){
			printf("\n");
		}
		printf("%d ", aArray[i]);
		fflush(stdout);
	}
}

/*
odd even sort
*/
int oddEven(int aID, int aValue, int aLen, int aSortingOrder,int aOrderType,int aMaster){
	MPI_Status status;
	bool working = true, ans;
	int value = aValue;
	int coord[INDICES];
	int masterId; //each row/col notify its master when done
	bool hasEverChanged = false;
	while (working){
		working = false;
		MPI_Cart_coords(cart_comm, aID, INDICES, coord);
		coord[aOrderType] = 0; //set master  coordinate(Relevant index)
		MPI_Cart_rank(cart_comm, coord, &masterId);
		value = iteration(aID, value, aLen, aSortingOrder, aOrderType, masterId);
		if (aID == masterId){
			int answers = 0;
			while (answers < aLen - 1){
				MPI_Recv(&ans, 1, MPI_C_BOOL, MPI_ANY_SOURCE, TAG_SLAVES_TO_MASTER, MPI_COMM_WORLD, &status);
				answers++;
				working = working || ans;
			}
			hasEverChanged = hasEverChanged || working;
		}
		//Master notifies each processor whether it needs to continue working or not
		MPI_Bcast(&working, 1, MPI_C_BOOL, masterId, MPI_COMM_WORLD);
	}
	if (aID == masterId && aID != aMaster){
		//Each master notify main master when its done
		MPI_Send(&hasEverChanged, 1, MPI_C_BOOL, aMaster, TAG_MASTERS_TO_MAIN_MASTER, MPI_COMM_WORLD);
	}
	return value;
}

/*
 iteration check both cases for even and odd iteraion at once
*/
int iteration(int aId, int aMyValue, int aLen, int aSortingOrder, int aOrderType, int aMasterId){
	int myValue = aMyValue, receivedValue = myValue;
	int currentIndex;
	MPI_Status status;
	int coord[INDICES];
	int  nextIndex = MPI_PROC_NULL, destId = MPI_PROC_NULL;
	//get current proccessor's coordinate
	MPI_Cart_coords(cart_comm, aId, INDICES, coord);
	//printf("\n iteration\n id = %d, (row,col) = (%d,%d) value = %d master = %d\n", aId, coord[0], coord[1], myValue, aMasterId);
	//fflush(stdout);
	currentIndex = coord[aOrderType]; //get correct location[i,j]
	bool isSwitched = false;
	//printf("\nbefore EVEN iteration\nindex = %d value = %d\n", currentIndex, myValue);
	//fflush(stdout);
	//P_even <--> P_odd
	if (currentIndex % 2 == 0) //even
	{
		if (currentIndex < aLen - 1) //check if last in row proc
			nextIndex = currentIndex + 1;
	}
	else //odd
	{
		if (currentIndex > 0) //check if first in row proc
			nextIndex = currentIndex - 1;
	}

	//get destination processor's id 
	if (nextIndex != MPI_PROC_NULL){
		coord[1 - aOrderType] = coord[1 - aOrderType];
		coord[aOrderType] = nextIndex;
		MPI_Cart_rank(cart_comm, coord, &destId);
	}
	
	//every 2 corresponded processors send each other their own value
	MPI_Sendrecv(&myValue, 1, MPI_INT, destId, TAG_SLAVES_TO_SLAVES, &receivedValue, 1, MPI_INT, destId,
		TAG_SLAVES_TO_SLAVES, MPI_COMM_WORLD, &status);

	//ascndeing and descending order
	myValue *= aSortingOrder;
	receivedValue *= aSortingOrder;
	if (currentIndex % 2 == 0)
	{
		if (myValue > receivedValue){
			myValue = receivedValue; //back to original value
			isSwitched = true;
		}
	}
	else
	{
		if (myValue< receivedValue){
			myValue = receivedValue ; //back to original value
			isSwitched = true;
		}
	}
	myValue *= aSortingOrder;
	/*printf("\nbefore ODD iteration\nindex = %d value = %d\n", currentIndex, myValue);
	fflush(stdout);*/
	receivedValue = myValue;
	nextIndex = MPI_PROC_NULL;
	destId = nextIndex;

	////ODD iteration \\\\
	//P_odd <--> P_even
	if (currentIndex % 2 == 0) //even
	{
		if (currentIndex > 0) //check if first in row proc
			nextIndex = currentIndex - 1;
	}
	else //odd
	{
		if (currentIndex < aLen - 1) //check if last in row proc
			nextIndex = currentIndex + 1;
	}

	//get destination processor's id 
	if (nextIndex != MPI_PROC_NULL){
		coord[1 - aOrderType] = coord[1 - aOrderType];
		coord[aOrderType] = nextIndex;
		MPI_Cart_rank(cart_comm, coord, &destId);
	}
	//every 2 corresponded processors send each other their own value
	MPI_Sendrecv(&myValue, 1, MPI_INT, destId, TAG_SLAVES_TO_SLAVES, &receivedValue, 1, MPI_INT, destId,
		TAG_SLAVES_TO_SLAVES, MPI_COMM_WORLD, &status);

	myValue *= aSortingOrder;
	receivedValue *= aSortingOrder;
	if (currentIndex % 2 == 0)
	{
		if (myValue < receivedValue){
			myValue = receivedValue; //back to original value
			isSwitched = true;
		}
	}
	else
	{
		if (myValue > receivedValue){
			myValue = receivedValue; //back to original value
			isSwitched = true;
		}
	}
	myValue *= aSortingOrder; 
	if (aId != aMasterId)
		MPI_Send(&isSwitched, 1, MPI_C_BOOL, aMasterId, TAG_SLAVES_TO_MASTER, MPI_COMM_WORLD);
	/*printf("\nafter all iteration : index = %d value = %d \n", currentIndex, myValue);
	fflush(stdout);*/
	return myValue;
}

/*
	shearsort algorithm
*/
int ShearSort(int aId, int aMainMaster, int aElementsPerRow, int aMyValue){
	int phase = 1;
	MPI_Status status;
	int coords[INDICES] = { 0, 0 };
	int sortingOrder = -1;
	bool shouldContinue = true;
	bool ans;
	int value = aMyValue;
	MPI_Cart_coords(cart_comm, aId, INDICES, coords);
	while (shouldContinue){
		shouldContinue = false;
		//get current proccessor's coordinate
		if (coords[ROW_INDEX] % 2 == 0){
			sortingOrder = ORDER_ASCENDING;
		}
		else{
			sortingOrder = ORDER_DESCENDING;
		}
	/*	printf("\n ROWS!!\n id = %d , [i,j] = [%d,%d], value = %d , sorting order = %d\n",aId,coords[0],coords[1],aMyValue,sortingOrder );
		fflush(stdout);*/

		//sort rows
		value = oddEven(aId, value, aElementsPerRow, sortingOrder, ORDER_ROW, aMainMaster);
		/*printf("\n FINISHED ROWS!!\n id = %d , [i,j] = [%d,%d], value = %d , sorting order = %d\n", aId, coords[0], coords[1], value, sortingOrder);
		fflush(stdout);*/
		if (aId == aMainMaster){
			int masters = 0;
			int maxCounter = aElementsPerRow - 1;
			/*printf("\n maxCounter = %d", maxCounter);
			fflush(stdout);*/
			//wait for all masters to notify when phase is done
			while (masters < maxCounter){
				MPI_Recv(&ans, 1, MPI_C_BOOL, MPI_ANY_SOURCE, TAG_MASTERS_TO_MAIN_MASTER, MPI_COMM_WORLD, &status);
				masters++;
				shouldContinue = shouldContinue || ans;
			}
		}
		MPI_Bcast(&shouldContinue, 1, MPI_C_BOOL, aMainMaster, MPI_COMM_WORLD);
		sortingOrder = ORDER_ASCENDING;
		/*printf("\n COL!!!!\n id = %d , [i,j] = [%d,%d], value = %d\n", aId, coords[0], coords[1], value);
		fflush(stdout);*/

		//sort coloumns
		value = oddEven(aId, value, aElementsPerRow, sortingOrder, ORDER_COL, aMainMaster);
		if (aId == aMainMaster){
			int masters = 0;
			int maxCounter = aElementsPerRow - 1;
			//wait for all masters to notify when phase is done
			while (masters < maxCounter){
				MPI_Recv(&ans, 1, MPI_C_BOOL, MPI_ANY_SOURCE, TAG_MASTERS_TO_MAIN_MASTER, MPI_COMM_WORLD, &status);
				masters++;
				shouldContinue = shouldContinue || ans;
			}
		}
		/*printf("\n COL FINISHED!!!!\n id = %d , [i,j] = [%d,%d], value = %d\n", aId, coords[0], coords[1], value);
		fflush(stdout);*/
		MPI_Bcast(&shouldContinue, 1, MPI_C_BOOL, aMainMaster, MPI_COMM_WORLD);
	}
	/*printf("\n DONE!!!!\n id = %d , [i,j] = [%d,%d], value = %d , sorting order = %d\n", aId, coords[0], coords[1], value, sortingOrder);
	fflush(stdout);*/
	return value;
}

