
class ArraySpec:
    """
    Data structure to hold the physical specification and analysis results
    for a specific array geometry.
    """
    def __init__(self, name: str, array_type: str, sensor_positions: list):
        # --- 1. Array Specification ---
        self.name = name                      # 1. Name of geometry
        self.array_type = array_type          # 2.1. array type (e.g., 'ULA', 'Nested')
        self.sensors_positions = sensor_positions # 2.3. sensors positions (list/numpy array of sensor locations)
        self.num_sensors = len(sensor_positions) # 2.2. Number of sensor
        self.sensor_spacing = None            # 2.4. sensor spacing (will be computed/set by subclass)

        # --- 2. Difference Coarray Data ---
        self.all_differences_table = None     # 3.1. Table: [s_i, s_j, n_i, n_j, (n_i-n_j)]
        self.all_differences_with_duplicates = None # 3.2. Array of all differences
        self.total_diff_computations = None   # 3.3. Total computation count
        self.unique_differences = None        # 3.4. Unique differences (virtual sensor positions)
        self.num_unique_positions = None      # 3.5. Number of unique positions
        self.physical_positions = None        # 3.6. Array of physical positions
        self.virtual_only_positions = None    # 3.7. Array of virtual-only positions
        self.coarray_positions = None         # 3.8. Coarray positions (physical + unique virtual)
        self.coarray_holes = None             # 3.9. All holes in coarray
        self.segmentation = None              # 3.10. Separate segmentation
        self.num_segmentation = None          # 3.11. Number of segmentation

        # --- 3. Weight Distribution ---
        self.weight_table = None              # 4.1. Table: [differences, count, sensor pairs]
        self.coarray_weight_distribution = None # 4.2. Array of weights

        # --- 4. Contiguous Segment Analysis ---
        self.all_contiguous_segments = None   # 5.1. List of all contiguous segments
        self.largest_contiguous_segment = None # 5.2. Largest contiguous segment
        self.shortest_contiguous_segment = None # 5.3. Shortest contiguous segment
        self.segment_lengths = None           # 5.4. Length of each segment
        self.max_detectable_sources = None    # 5.5. Maximum detectable sources
        self.segment_ranges = None            # 5.6. Range of each segment

        # --- 5. Holes Analysis ---
        self.missing_virtual_positions = None # 6.1. Missing position of virtual array
        self.num_holes = None                 # 6.2. Number of holes

        # --- 6. Performance Summary ---
        self.performance_summary_table = None # 7. Performance summary table


from abc import ABC, abstractmethod

class BaseArrayProcessor(ABC):
    """
    Abstract Base Class for all MIMO array geometry processors.
    This structure ensures all derived classes implement the necessary
    steps for geometry, coarray, weight, and performance analysis.
    """
    def __init__(self, name: str, array_type: str, sensor_positions: list):
        self.data = ArraySpec(name, array_type, sensor_positions)

    # --- Core Geometry Implementation (2. Physical Array Specification) ---
    @abstractmethod
    def compute_array_spacing(self):
        """Computes and sets self.data.sensor_spacing."""
        pass

    # --- Coarray Computation (3. Difference Coarray Computation) ---
    @abstractmethod
    def compute_all_differences(self):
        """
        Computes all difference elements and populates:
        3.1. self.data.all_differences_table
        3.2. self.data.all_differences_with_duplicates
        3.3. self.data.total_diff_computations
        """
        pass

    @abstractmethod
    def analyze_coarray(self):
        """
        Processes the differences to find unique elements, holes, and segments:
        3.4. self.data.unique_differences
        3.5. self.data.num_unique_positions
        3.6. self.data.physical_positions
        3.7. self.data.virtual_only_positions
        3.8. self.data.coarray_positions
        3.9. self.data.coarray_holes
        3.10. self.data.segmentation
        3.11. self.data.num_segmentation
        """
        pass

    @abstractmethod
    def plot_coarray(self):
        """
        3.12. Represents physical, virtual-only and coarray position 
        with specific labels and proper legend.
        """
        pass

    # --- Weight Distribution (4. Weight Distribution) ---
    @abstractmethod
    def compute_weight_distribution(self):
        """
        Calculates the weight (count) of each unique lag:
        4.1. self.data.weight_table
        4.2. self.data.coarray_weight_distribution
        """
        pass

    # --- Contiguous Segment Analysis (5. Contiguous Segment Analysis) ---
    @abstractmethod
    def analyze_contiguous_segments(self):
        """
        Identifies and analyzes contiguous segments:
        5.1. self.data.all_contiguous_segments
        5.2. self.data.largest_contiguous_segment
        5.3. self.data.shortest_contiguous_segment
        5.4. self.data.segment_lengths
        5.5. self.data.max_detectable_sources
        5.6/5.7. self.data.segment_ranges
        """
        pass
    
    # --- Holes Analysis (6. Holes Analysis) ---
    @abstractmethod
    def analyze_holes(self):
        """
        Identifies and counts holes:
        6.1. self.data.missing_virtual_positions
        6.2. self.data.num_holes
        """
        pass

    # --- Performance Summary (7. Performance Summary) ---
    @abstractmethod
    def generate_performance_summary(self):
        """
        Generates the final performance summary table:
        7. self.data.performance_summary_table
        """
        pass
    
    def run_full_analysis(self):
        """
        Executes all analysis steps in a logical sequence.
        """
        print(f"--- Starting analysis for {self.data.name} ({self.data.array_type}) ---")
        self.compute_array_spacing()
        self.compute_all_differences()
        self.analyze_coarray()
        self.compute_weight_distribution()
        self.analyze_contiguous_segments()
        self.analyze_holes()
        self.generate_performance_summary()
        self.plot_coarray() # Optional plotting step
        print("--- Analysis Complete ---")
        return self.data