# Phase 1: The Riemann Data-Verse

**Objective:** Create the world's most comprehensive, high-precision, and accessible database of the non-trivial zeros of the Riemann zeta function and related L-functions. This dataset is the empirical ground truth for all subsequent machine learning models.

This phase focuses on:
1.  **Calculating Zeros:** Implementing and scaling algorithms (like Odlyzko-Schönhage) for rapid, high-precision zero computation.
2.  **Storing Zeros:** Utilizing a high-performance time-series database capable of handling billions of data points.
3.  **Accessing Zeros:** Providing a public API for querying and downloading zero data.
4.  **Consolidating Data:** The long-term goal is to integrate and standardize existing datasets from sources like Andrew Odlyzko's tables, the ZetaGrid project, and the LMFDB (L-functions and Modular Forms Database).

## Components

1.  **`calculator/`**: Contains the C++ code for zero computation and Python wrappers.
    *   `zeta_calculator.cpp`: A high-performance C++ executable for calculating zeta zeros using the Odlyzko-Schönhage algorithm (currently a stub). It's intended to use high-precision arithmetic libraries like GMP/MPFR.
    *   `wrapper.py`: A Python interface (using ctypes or pybind11) for the C++ calculator, allowing easier integration with other Python-based components.
    *   `Makefile` (or `CMakeLists.txt` - not generated yet): Build scripts for compiling the C++ calculator into an executable and/or a shared library for the Python wrapper.

2.  **`api/`**: Contains the FastAPI application for serving zero data.
    *   `main.py`: The main FastAPI application file. It defines endpoints for querying zero counts, ranges of zeros, and potentially for ingesting new zeros.

3.  **`docker-compose.yml`**: A Docker Compose file to set up and manage the database service (e.g., ClickHouse or QuestDB) and potentially the API service.

## Technology Stack

*   **Computation:** C++, Python
*   **High-Precision Libraries:** GMP/MPFR, Boost.Multiprecision (for C++). Python's `mpmath` library can be used for validation and some Python-side calculations.
*   **Database:** ClickHouse or QuestDB (ClickHouse is configured in the provided `docker-compose.yml`). These are chosen for their performance with massive time-series data.
*   **API:** Python with FastAPI.

## Getting Started

### 1. Build the Calculator (C++)

Navigate to the `calculator/` directory. You'll need a C++ compiler (like g++) and the GMP library installed.
A `Makefile` or `CMakeLists.txt` would typically be provided here. For the current stub:

**Example Compilation (Linux/macOS with g++ and GMP):**

To build the executable:
```bash
g++ zeta_calculator.cpp -o zeta_calculator -lgmp -std=c++11
```

To build a shared library for the Python `ctypes` wrapper (example for Linux):
```bash
g++ -shared -o zeta_calculator_lib.so -fPIC zeta_calculator.cpp -lgmp -std=c++11
```
(On macOS, the extension would be `.dylib` and you might need different flags.)

Then run: `./zeta_calculator <start_t> <end_t> [precision_bits]`

### 2. Launch the Database Service

A `docker-compose.yml` file is provided to run a ClickHouse database instance using Docker.
Ensure you have Docker and Docker Compose installed.

From the `data_verse/` directory, run:
```bash
docker-compose up -d clickhouse-server
```
This will start a ClickHouse server in detached mode. The database will be accessible on the host at the ports specified in `docker-compose.yml` (default HTTP: 8123, TCP: 9000).

To stop the database:
```bash
docker-compose down
```

**Database Schema:**
The `docker-compose.yml` includes comments suggesting a possible table schema for storing zeros in ClickHouse. This schema should be created after the database is running, typically via the API's ingestion logic or an initialization script.

Example Table: `zeta_zeros_db.zeros`
Columns might include: `id`, `height_t_str` (string for full precision), `precision_bits`, `source`, `computation_timestamp`, etc.

### 3. Run the API Service

The API is a FastAPI application located in `api/main.py`.

**Prerequisites:**
Install Python dependencies:
```bash
pip install fastapi uvicorn "clickhouse-driver" # or "clickhouse-connect"
```

**Running the API:**
Navigate to the `api/` directory (or ensure `main.py` is in your Python path).
Run using Uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`. Interactive documentation (Swagger UI) will be at `http://localhost:8000/docs`.

The API (currently using a mock database) provides endpoints like:
*   `/zeros/count`: Get the total count of zeros.
*   `/zeros/range?start_height_t=...&end_height_t=...&limit=...`: Get zeros within a specified height range.
*   `/zeros/ingest`: (POST) Ingest new zero data.

### 4. Using the Python Wrapper for the Calculator

The `calculator/wrapper.py` script demonstrates how to call the C++ `calculate_zeros_batch_c` function from Python using `ctypes`.
Ensure the shared library (`zeta_calculator_lib.so` or similar) is compiled and accessible.

Run the wrapper script (from the `calculator/` directory, assuming the library is there):
```bash
python wrapper.py
```
This will attempt to load the C++ library and call the example function.

## Development Notes

*   **Odlyzko-Schönhage Algorithm:** The C++ `zeta_calculator.cpp` currently contains only a high-level stub for this complex algorithm. Full implementation is a major research and engineering task.
*   **High-Precision Data Transfer:** Transferring arbitrary-precision numbers between C++ and Python (and storing/retrieving them via the API and database) requires careful handling to avoid precision loss. Storing as strings is a common strategy for full fidelity.
*   **Distributed Computation:** The current scaffolding does not include a full distributed computation management system. This would be a significant extension, potentially involving job queues and worker management to scale the zero computation.
*   **Database Client:** The `api/main.py` uses a mock database client. For a real deployment, it should be configured to connect to the live ClickHouse (or QuestDB) instance, using libraries like `clickhouse-driver` or `clickhouse-connect`.
*   **Error Handling & Logging:** Production-quality code would require more robust error handling, input validation, and comprehensive logging across all components.

This phase lays the critical data foundation upon which all other analytical and discovery phases of Project ZetaFlow will be built.
