#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define POPULATION_SIZE 20
#define CHROMOSOME_LENGTH 3

typedef struct Individual {
    int chromosome[CHROMOSOME_LENGTH];
    int fitness;
    struct Individual* next;
} Individual;

Individual* create_individual() {
    Individual* ind = (Individual*)malloc(sizeof(Individual));
    for (int i = 0; i < CHROMOSOME_LENGTH; i++) {
        ind->chromosome[i] = rand() % 2;
    }
    ind->fitness = 0;
    ind->next = NULL;
    return ind;
}

int calculate_fitness(int* chromosome) {
    int ones_count = 0;
    for (int i = 0; i < CHROMOSOME_LENGTH; i++) {
        if (chromosome[i] == 1) {
            ones_count++;
        }
    }
    return ones_count;
}

void insert_sorted(Individual** head, Individual* new_ind) {
    if (*head == NULL || new_ind->fitness >= (*head)->fitness) {
        new_ind->next = *head;
        *head = new_ind;
    } else {
        Individual* current = *head;
        while (current->next != NULL && current->next->fitness > new_ind->fitness) {
            current = current->next;
        }
        new_ind->next = current->next;
        current->next = new_ind;
    }
}

void print_individual(Individual* ind) {
    for (int i = 0; i < CHROMOSOME_LENGTH; i++) {
        printf("%d", ind->chromosome[i]);
    }
    printf(", Fitness = %d\n", ind->fitness);
}

void print_population(Individual* head) {
    Individual* current = head;
    int count = 0;
    while (current != NULL && count < POPULATION_SIZE) {
        printf("Individual %d: ", count + 1);
        print_individual(current);
        current = current->next;
        count++;
    }
    printf("...\n");
}

void create_offspring(Individual* parentA, Individual* parentB) {
    int i = 0, j = CHROMOSOME_LENGTH;

    while (j > 1) {
        // Create two offspring
        Individual* offspring1 = create_individual();  // Temporary
        Individual* offspring2 = create_individual();  // Temporary
        
        // Offspring 1: A[i] + B[j]
        for (int k = 0; k < CHROMOSOME_LENGTH; k++) {
            offspring1->chromosome[k] = (k <= i) ? parentA->chromosome[k] : parentB->chromosome[k];
        }

        // Offspring 2: B[i] + A[j]
        for (int k = 0; k < CHROMOSOME_LENGTH; k++) {
            offspring2->chromosome[k] = (k <= i) ? parentB->chromosome[k] : parentA->chromosome[k];
        }

        // Calculate fitness
        offspring1->fitness = calculate_fitness(offspring1->chromosome);
        offspring2->fitness = calculate_fitness(offspring2->chromosome);

        // Print offspring
        printf("\nOffspring 1 (A[%d] + B[%d]): ", i, j - 1);
        print_individual(offspring1);

        printf("Offspring 2 (B[%d] + A[%d]): ", i, j - 1);
        print_individual(offspring2);

        // Find the two largest fitness values
        int values[4] = {parentA->fitness, parentB->fitness, offspring1->fitness, offspring2->fitness};
        int largest = -1, secondLargest = -1;

        for (int k = 0; k < 4; k++) {
            if (values[k] > largest) {
                secondLargest = largest;
                largest = values[k];
            } else if (values[k] > secondLargest) {
                secondLargest = values[k];
            }
        }

        // Store the largest in parentA->fitness and second-largest in parentB->fitness
        parentA->fitness = largest;
        parentB->fitness = secondLargest;

        i++;
        j--;
    }
}



int main() {
    srand(time(NULL)); // Ensure randomness

    Individual* population = NULL;

    for (int i = 0; i < POPULATION_SIZE; i++) {
        Individual* new_ind = create_individual();
        new_ind->fitness = calculate_fitness(new_ind->chromosome);
        insert_sorted(&population, new_ind);
    }

    printf("Initial population:\n");
    print_population(population);

    int generation = 1;
    Individual* parentA;
    Individual* parentB;
    int a , b = 0;

    // for(int i =0 ; i<3; i++)
    while (1)
     {
       
        parentA = population;        // Best individual (highest fitness)
        parentB = population->next;  // Second-best individual

        a = parentB->fitness;
        b = parentB->fitness;

        printf("\nGeneration %d\n", generation);
        printf("\nSelected Best Parents:\n");
        printf("Parent A: ");
        print_individual(parentA);
        printf("Parent B: ");
        print_individual(parentB);

        if (parentA->fitness == CHROMOSOME_LENGTH || parentB->fitness == CHROMOSOME_LENGTH) {
            printf("\nMax Fitness Reached! Ending Process.\n\n");
            break;
        }
        

        // Create offspring but don't store them
        create_offspring(parentA, parentB);

        

        if(parentB->fitness == b && parentA->fitness == a){
             printf("\nGeneration %d\n", generation);
        printf("\nSelected Best Parents:\n");
        printf("Parent A: ");
        print_individual(parentA);
        printf("Parent B: ");
        print_individual(parentB);
            printf("\nEnding Process.");
            break;
        }

        


        generation++;
    }
}
