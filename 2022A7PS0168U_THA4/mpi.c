#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>

#define MAX_LINE_LENGTH 1024
#define MAX_EVENT_LENGTH 256
#define MAX_EVENTS 100000  // Adjust based on expected number of unique events

// Structure to store event and its count
typedef struct {
    char event[MAX_EVENT_LENGTH];
    int count;
} EventCount;

// Structure to hold the result across all processes
typedef struct {
    char event[MAX_EVENT_LENGTH];
    int count;
} GlobalEventCount;

// Function to extract event from a log line
void extract_event(char *line, char *event) {
    // Assuming the event is the first field in the log line
    // Modify this function based on your log format
    char *token = strtok(line, " \t");
    if (token != NULL) {
        strncpy(event, token, MAX_EVENT_LENGTH - 1);
        event[MAX_EVENT_LENGTH - 1] = '\0';
    } else {
        event[0] = '\0';
    }
}

// Compare function for qsort to sort events by count in descending order
int compare_events(const void *a, const void *b) {
    const GlobalEventCount *event_a = (const GlobalEventCount *)a;
    const GlobalEventCount *event_b = (const GlobalEventCount *)b;
    return event_b->count - event_a->count;
}

int main(int argc, char *argv[]) {
    int rank, size, provided;
    MPI_File fh_in, fh_out1, fh_out2;
    MPI_Status status;
    MPI_Offset filesize, local_start, local_end, offset;
    
    // Initialize MPI with thread support
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Check if input file is provided
    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <input_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    char *filename = argv[1];
    
    // Open input file with MPI-IO
    if (MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_in) != MPI_SUCCESS) {
        if (rank == 0) {
            printf("Error opening file %s\n", filename);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Get file size
    MPI_File_get_size(fh_in, &filesize);
    
    // Calculate portion for each process
    local_start = rank * filesize / size;
    local_end = (rank + 1) * filesize / size;
    
    // Adjust start to avoid splitting lines (except for rank 0)
    if (rank > 0) {
        char ch;
        offset = local_start;
        
        // Look backward for a newline
        do {
            offset--;
            MPI_File_read_at(fh_in, offset, &ch, 1, MPI_CHAR, &status);
        } while (offset > 0 && ch != '\n');
        
        // If newline found, start after it, otherwise start at beginning of file
        if (ch == '\n') {
            local_start = offset + 1;
        } else {
            local_start = 0;
        }
    }
    
    // Adjust end to avoid splitting lines (except for the last process)
    if (rank < size - 1) {
        char ch;
        offset = local_end - 1;
        
        // Look forward for a newline
        do {
            offset++;
            MPI_File_read_at(fh_in, offset, &ch, 1, MPI_CHAR, &status);
        } while (offset < filesize && ch != '\n');
        
        // If newline found, end at it, otherwise end at end of file
        if (ch == '\n') {
            local_end = offset + 1;
        } else {
            local_end = filesize;
        }
    } else {
        local_end = filesize;
    }
    
    // Calculate local buffer size
    MPI_Offset local_size = local_end - local_start;
    
    // Allocate buffer for local portion of file
    char *buffer = (char *)malloc((local_size + 1) * sizeof(char));
    if (buffer == NULL) {
        printf("Process %d: Memory allocation failed\n", rank);
        MPI_File_close(&fh_in);
        MPI_Finalize();
        return 1;
    }
    
    // Read local portion of file
    MPI_File_read_at(fh_in, local_start, buffer, local_size, MPI_CHAR, &status);
    buffer[local_size] = '\0';
    
    // Hash map to count events locally
    EventCount *local_events = (EventCount *)malloc(MAX_EVENTS * sizeof(EventCount));
    if (local_events == NULL) {
        printf("Process %d: Memory allocation for events failed\n", rank);
        free(buffer);
        MPI_File_close(&fh_in);
        MPI_Finalize();
        return 1;
    }
    
    int local_unique_events = 0;
    
    // Process buffer line by line using OpenMP
    #pragma omp parallel
    {
        char *line_start, *saveptr;
        char line[MAX_LINE_LENGTH];
        char event[MAX_EVENT_LENGTH];
        
        // Make a thread-local copy of the buffer for strtok_r
        char *local_buffer = strdup(buffer);
        if (local_buffer == NULL) {
            #pragma omp critical
            {
                printf("Thread in Process %d: Memory allocation failed\n", rank);
            }
        } else {
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < 1; i++) {  // Dummy loop for OpenMP
                for (line_start = strtok_r(local_buffer, "\n", &saveptr); 
                     line_start != NULL; 
                     line_start = strtok_r(NULL, "\n", &saveptr)) {
                    
                    strncpy(line, line_start, MAX_LINE_LENGTH - 1);
                    line[MAX_LINE_LENGTH - 1] = '\0';
                    
                    // Extract event from line
                    extract_event(line, event);
                    
                    if (strlen(event) > 0) {
                        // Update event count in thread-safe manner
                        #pragma omp critical
                        {
                            int found = 0;
                            for (int j = 0; j < local_unique_events; j++) {
                                if (strcmp(local_events[j].event, event) == 0) {
                                    local_events[j].count++;
                                    found = 1;
                                    break;
                                }
                            }
                            
                            if (!found && local_unique_events < MAX_EVENTS) {
                                strcpy(local_events[local_unique_events].event, event);
                                local_events[local_unique_events].count = 1;
                                local_unique_events++;
                            }
                        }
                    }
                }
            }
            free(local_buffer);
        }
    }
    
    // Free buffer as it's no longer needed
    free(buffer);
    MPI_File_close(&fh_in);
    
    // Gather counts of unique events from all processes
    int *event_counts = (int *)malloc(size * sizeof(int));
    if (event_counts == NULL) {
        printf("Process %d: Memory allocation for event counts failed\n", rank);
        free(local_events);
        MPI_Finalize();
        return 1;
    }
    
    // Gather the number of unique events from each process
    MPI_Allgather(&local_unique_events, 1, MPI_INT, event_counts, 1, MPI_INT, MPI_COMM_WORLD);
    
    // Calculate total unique events and displacements
    int total_events = 0;
    int *displs = (int *)malloc(size * sizeof(int));
    if (displs == NULL) {
        printf("Process %d: Memory allocation for displacements failed\n", rank);
        free(local_events);
        free(event_counts);
        MPI_Finalize();
        return 1;
    }
    
    for (int i = 0; i < size; i++) {
        displs[i] = total_events;
        total_events += event_counts[i];
    }
    
    // Create MPI datatype for EventCount
    MPI_Datatype event_type;
    int blocklengths[2] = {MAX_EVENT_LENGTH, 1};
    MPI_Aint displacements[2];
    MPI_Datatype types[2] = {MPI_CHAR, MPI_INT};
    
    // Get the address offsets of EventCount structure members
    EventCount temp;
    MPI_Aint base_address;
    MPI_Get_address(&temp, &base_address);
    MPI_Get_address(&temp.event, &displacements[0]);
    MPI_Get_address(&temp.count, &displacements[1]);
    
    // Make relative to the base address
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
    
    // Create and commit the MPI datatype
    MPI_Type_create_struct(2, blocklengths, displacements, types, &event_type);
    MPI_Type_commit(&event_type);
    
    // Allocate memory for all events
    GlobalEventCount *all_events = NULL;
    if (rank == 0) {
        all_events = (GlobalEventCount *)malloc(total_events * sizeof(GlobalEventCount));
        if (all_events == NULL) {
            printf("Process 0: Memory allocation for all events failed\n");
            free(local_events);
            free(event_counts);
            free(displs);
            MPI_Type_free(&event_type);
            MPI_Finalize();
            return 1;
        }
    }
    
    // Gather all events to rank 0
    MPI_Gatherv(local_events, local_unique_events, event_type, 
                all_events, event_counts, displs, event_type, 
                0, MPI_COMM_WORLD);
    
    // Free local events as they're no longer needed
    free(local_events);
    free(event_counts);
    free(displs);
    MPI_Type_free(&event_type);
    
    // Process 0 merges and sorts events
    if (rank == 0) {
        // Merge duplicate events
        GlobalEventCount *merged_events = (GlobalEventCount *)malloc(total_events * sizeof(GlobalEventCount));
        if (merged_events == NULL) {
            printf("Process 0: Memory allocation for merged events failed\n");
            free(all_events);
            MPI_Finalize();
            return 1;
        }
        
        int merged_count = 0;
        
        for (int i = 0; i < total_events; i++) {
            int found = 0;
            for (int j = 0; j < merged_count; j++) {
                if (strcmp(all_events[i].event, merged_events[j].event) == 0) {
                    merged_events[j].count += all_events[i].count;
                    found = 1;
                    break;
                }
            }
            
            if (!found) {
                strcpy(merged_events[merged_count].event, all_events[i].event);
                merged_events[merged_count].count = all_events[i].count;
                merged_count++;
            }
        }
        
        // Sort merged events by count in descending order
        qsort(merged_events, merged_count, sizeof(GlobalEventCount), compare_events);
        
        // Open output files for writing
        if (MPI_File_open(MPI_COMM_SELF, "event_count.txt", 
                         MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_out1) != MPI_SUCCESS) {
            printf("Error opening output file event_count.txt\n");
            free(all_events);
            free(merged_events);
            MPI_Finalize();
            return 1;
        }
        
        if (MPI_File_open(MPI_COMM_SELF, "top_10.txt", 
                         MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_out2) != MPI_SUCCESS) {
            printf("Error opening output file top_10.txt\n");
            MPI_File_close(&fh_out1);
            free(all_events);
            free(merged_events);
            MPI_Finalize();
            return 1;
        }
        
        // Write all events and their counts to event_count.txt
        char line[MAX_LINE_LENGTH];
        MPI_Offset write_offset = 0;
        
        for (int i = 0; i < merged_count; i++) {
            int len = snprintf(line, MAX_LINE_LENGTH, "%s: %d\n", 
                              merged_events[i].event, merged_events[i].count);
            MPI_File_write_at(fh_out1, write_offset, line, len, MPI_CHAR, &status);
            write_offset += len;
        }
        
        // Write top 10 events to top_10.txt
        write_offset = 0;
        int top_count = (merged_count < 10) ? merged_count : 10;
        
        for (int i = 0; i < top_count; i++) {
            int len = snprintf(line, MAX_LINE_LENGTH, "%s: %d\n", 
                              merged_events[i].event, merged_events[i].count);
            MPI_File_write_at(fh_out2, write_offset, line, len, MPI_CHAR, &status);
            write_offset += len;
        }
        
        // Close output files
        MPI_File_close(&fh_out1);
        MPI_File_close(&fh_out2);
        
        // Free memory
        free(merged_events);
        free(all_events);
        
        printf("Processing complete. Results written to event_count.txt and top_10.txt\n");
    }
    
    MPI_Finalize();
    return 0;
}