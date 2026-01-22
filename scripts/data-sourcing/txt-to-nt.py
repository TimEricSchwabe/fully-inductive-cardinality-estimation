#!/usr/bin/env python3
"""
Convert multiple txt files containing RDF triples to a single N-Triples (.nt) file.
Each line in input files should have format: subject predicate object (tab-separated)
"""

# ==================== CONFIGURATION ====================
# Input txt files to process
INPUT_FILES = [
    "DB100K_train.txt",
    "DB100K_valid.txt",
    "DB100K_test.txt",
]

# Output N-Triples file
OUTPUT_FILE = "data/DBpedia100k/raw/DBpedia100k.nt"

# Base URI for wrapping entities
BASE_URI = "http://example.org"
# =======================================================


def convert_to_nt(input_files, output_file, base_uri):
    """
    Convert multiple txt files with triples to a single N-Triples file.
    
    Args:
        input_files: List of paths to input txt files
        output_file: Path to output .nt file
        base_uri: Base URI to use for wrapping entities
    """
    total_triples = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            print(f"Processing: {input_file}")
            file_triples = 0
            
            try:
                with open(input_file, 'r', encoding='utf-8') as in_f:
                    for line_num, line in enumerate(in_f, 1):
                        line = line.strip()
                        
                        # Skip empty lines
                        if not line:
                            continue
                        
                        # Split by tab or whitespace
                        parts = line.split('\t') if '\t' in line else line.split()
                        
                        if len(parts) != 3:
                            print(f"  Warning: Line {line_num} doesn't have 3 parts, skipping: {line}")
                            continue
                        
                        subject, predicate, obj = parts
                        
                        # Format as N-Triples: <uri_s> <uri_p> <uri_o> .
                        nt_line = f"<{base_uri}{subject}> <{base_uri}{predicate}> <{base_uri}{obj}> .\n"
                        out_f.write(nt_line)
                        
                        file_triples += 1
                        total_triples += 1
                
                print(f"  Processed {file_triples} triples")
                
            except FileNotFoundError:
                print(f"  Error: File not found - {input_file}")
            except Exception as e:
                print(f"  Error processing file: {e}")
    
    print(f"\nTotal triples written: {total_triples}")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    convert_to_nt(INPUT_FILES, OUTPUT_FILE, BASE_URI)

