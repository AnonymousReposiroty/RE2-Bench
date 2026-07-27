# Type Extractor

This directory contains type extraction scripts that process raw data collected from various benchmark projects to generate master JSON files. These master JSON files are required for all subsequent processing steps in the RE2-Bench pipeline.

## Overview

After collecting raw data using the specific data generation pipeline, you must run the type extractor to generate master JSON files from the raw data. The type extractor modifies the raw data in-place by adding type information to each test case.

## Scripts

### 1. SWE-Bench Type Extraction

**Script**: `extract_swe_bench_types.sh`

Extracts types for the SWE-Bench dataset.

```bash
./extract_swe_bench_types.sh
```

**What it does**:
- Processes raw JSONL files from `../../dataset/re2-bench/input-output/Swebench`
- Adds type information to input and output values
- Modifies the original files in-place

### 2. Non-SWE-Bench Type Extraction

**Script**: `extract_non_swe_bench_types.sh`

Extracts types for all non-SWE-Bench projects used in the benchmark.

```bash
./extract_non_swe_bench_types.sh
```

**What it does**:
- Processes multiple projects: HumanEval, ClassEval, CruxEval, and Avatar
- Iterates through each project directory in `../../dataset/re2-bench/input-output/{project}`
- Adds comprehensive type information including function parameters, return types, and class properties
- Modifies the original files in-place

## Supported Projects

### SWE-Bench
- **Path**: `dataset/re2-bench/input-output/Swebench`
- **Processing**: Extracts types for function inputs and return values

### Non-SWE-Bench Projects
- **HumanEval**: Function-based problems with input/output type extraction
- **ClassEval**: Class-based problems with method and property type extraction
- **CruxEval**: Complex algorithmic problems with type analysis
- **Avatar**: Competitive programming problems with input/output type extraction

## Important Notes

⚠️ **Warning**: These scripts modify the raw data files in-place. Make sure you have backups if needed.

- The scripts do not generate new output files
- They enhance existing JSONL files by adding type information
- All subsequent pipeline steps depend on these enhanced master JSON files
- The type extraction must be completed before running validation or analysis scripts

## Usage Workflow

1. **Collect Raw Data**: Use the data generation pipeline to collect raw test cases
2. **Extract Types**: Run the appropriate type extraction script(s)
3. **Continue Pipeline**: Use the enhanced master JSON files for validation and analysis
