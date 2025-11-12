#!/usr/bin/env python3
# Author: Jingmei Yang (jmyang@bu.edu)
"""
PubMed Text Preprocessing Pipeline


Main functionality:
1. Loads PubMed article data from JSONL files
2. Applies multi-stage text preprocessing pipeline
3. Removes citations, HTML tags, and unwanted sections
4. Standardizes text formatting and encoding
5. Outputs cleaned data with word count statistics

"""


import re
from html import unescape
import os
import json
from log_info import setup_logger
import argparse
from datasets import load_dataset

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def remove_citations(text):
    """
    Remove academic citations from text in both square brackets and parentheses format.
    
    Handles various citation formats commonly found in biomedical literature:
    - Single citations: [5], (3)
    - Citation ranges: [5-9], (6-10)
    - Multiple citations: [5, 6, 7], (1, 3, 5)
    
    Args:
        text (str): Input text containing citations
        
    Returns:
        str: Text with citations removed
    """
    # Remove numeric citations and ranges in square brackets
    text = re.sub(r'\[\s*\d+\s*(?:–|-)\s*\d+\s*\]', '', text)  # Ranges like [5-9] or [ 5 - 9 ]
    text = re.sub(r'\[\s*(?:\d+\s*,\s*)*\d+\s*\]', '', text)  # Single numbers and lists like [5, 6, 7]

    # Remove numeric citations and ranges in parentheses
    text = re.sub(r'\(\s*\d+\s*(?:–|-)\s*\d+\s*\)', '', text)  # Ranges like (6-9) or ( 6 - 9 )
    text = re.sub(r'\(\s*(?:\d+\s*,\s*)*\d+\s*\)', '', text)  # Single numbers and lists like (6, 7, 9)

    return text


def remove_unwanted_sections(text):
    """
    Remove standard academic paper sections that don't contribute to main content.
    
    Removes sections typically found at the end of research papers that contain
    metadata rather than scientific content, including funding information,
    author contributions, and supplementary material references.
    
    Args:
        text (str): Full text of the research paper
        
    Returns:
        str: Text with unwanted sections removed (truncated at first match)
    """
    # Define patterns for sections commonly found at end of academic papers
    section_patterns = [
        r"competing interest(s)?",
        r"funding",
        r"author('s)? contribution(s)?",
        r"disclosure(s)?",
        r"declaration(s)?",
        r"acknowledgement(s)?",
        r"supplementary data",
        r"supporting information",
        r"supplemental material"
    ]

    # Combine all patterns into a single regex, matching any of them
    combined_pattern = r"(" + "|".join(section_patterns) + ")"

    # Find the first occurrence of any unwanted section
    match = re.search(combined_pattern, text, flags=re.IGNORECASE)

    if match:
        # Truncate text at the start of the first unwanted section
        cleaned_text = text[:match.start()]
    else:
        # No unwanted sections found, return original text
        cleaned_text = text

    return cleaned_text


def remove_html_tags(text):
    """
    Remove all HTML/XML tags from text while preserving content.
    
    Uses regex to match and remove all HTML/XML tag patterns,
    commonly found in web-scraped or formatted academic content.
    
    Args:
        text (str): Text potentially containing HTML/XML tags
        
    Returns:
        str: Text with all HTML/XML tags removed
    """
    # Regular expression to match HTML/XML tags
    tag_pattern = re.compile(r'<.*?>')
    cleaned_text = re.sub(tag_pattern, '', text)
    return cleaned_text


def convert_html_entitie2characters(text):
    """
    Convert HTML entities to their corresponding characters.
    
    Converts HTML entities like &amp;, &lt;, &gt;, &#39; etc. to their
    actual character representations for proper text processing.
    
    Args:
        text (str): Text containing HTML entities
        
    Returns:
        str: Text with HTML entities converted to characters
    """
    cleaned_text = unescape(text)
    return cleaned_text


def remove_whitespace(text):
    """
    Normalize whitespace by collapsing multiple spaces and trimming.
    
    Replaces any sequence of whitespace characters (spaces, tabs, newlines)
    with a single space and removes leading/trailing whitespace.
    
    Args:
        text (str): Text with potentially irregular whitespace
        
    Returns:
        str: Text with normalized whitespace
    """
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text


def decode_text(text):
    """
    Decode Unicode escape sequences in text with error handling.
    
    Attempts to decode Unicode escape sequences that may be present in
    the input text, with graceful error handling for malformed sequences.
    
    Args:
        text (str): Text potentially containing Unicode escape sequences
        
    Returns:
        str: Text with Unicode sequences decoded
    """
    try:
        text = text.encode().decode('unicode_escape', errors='replace')
    except UnicodeDecodeError as e:
        logger.info(f"Found error - {e} in text")
    return text


def preprocess_text_pipeline(text):
    """
    Apply the complete text preprocessing pipeline.
    
    Executes all preprocessing steps in sequence to clean and standardize
    biomedical text for machine learning applications. The pipeline order
    is optimized to handle interdependent cleaning operations effectively.
    
    Pipeline steps:
    1. Decode Unicode escape sequences
    2. Remove HTML/XML tags
    3. Convert HTML entities to characters
    4. Remove academic citations
    5. Remove unwanted paper sections
    6. Normalize whitespace
    
    Args:
        text (str): Raw text from PubMed article
        
    Returns:
        str: Fully preprocessed and cleaned text
    """
    text = decode_text(text)
    text = remove_html_tags(text)
    text = convert_html_entitie2characters(text)
    text = remove_citations(text)
    text = remove_unwanted_sections(text)
    text = remove_whitespace(text)
    return text


def preprocess_example(batch):
    """
    Process a batch of examples through the preprocessing pipeline.
    
    Applies the preprocessing pipeline to a batch of texts and calculates
    word counts for each processed text. Designed for use with HuggingFace
    datasets batch processing.
    
    Args:
        batch (dict): Batch of examples with 'text' field
        
    Returns:
        dict: Batch with processed texts and added word_count field
    """
    # Apply preprocessing pipeline to each text in the batch
    processed_texts = [preprocess_text_pipeline(text) for text in batch["text"]]
    
    # Update batch with processed texts
    batch["text"] = processed_texts
    
    # Calculate and add word counts for each processed text
    batch["word_count"] = [len(text.split()) for text in processed_texts]
    
    return batch


def save_to_jsonl(dataset, filename, mode='a'):
    """
    Save dataset to JSONL format file.
    
    Writes each example in the dataset as a separate JSON object on its own line,
    following the JSONL (JSON Lines) format commonly used for large datasets.
    
    Args:
        dataset: Iterable dataset to save
        filename (str): Output file path
        mode (str): File open mode ('a' for append, 'w' for write)
    """
    with open(filename, mode) as f:
        for item in dataset:
            # Convert each item to JSON and write on separate line
            json_record = json.dumps(item)
            f.write(json_record + '\n')


def get_args():
    """
    Parse command line arguments for the preprocessing script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='PubMed_Preprocessor')
    parser.add_argument('--disease', type=str, default='Your Disease',
                        help='Specify the disease name (default: Your Disease)')
    parser.add_argument('--outfile', type=str, default='preprocessed_data.jsonl', 
                        help='Specify the output file (default: preprocessed_data.jsonl)')
    parser.add_argument('--infile', type=str, default='data.jsonl',
                        help='Specify the input file (default: data.jsonl)')
    parser.add_argument('--outdir', type=str, default="/data/Your Path/", 
                        help='Specify the output directory (default: /data/Your Path/)')
    parser.add_argument('--indir', type=str, default="/data/Your Path/", 
                        help='Specify the input directory (default: /data/Your Path/)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    Main execution pipeline for PubMed text preprocessing.
    
    Workflow:
    1. Parse command line arguments
    2. Set up logging
    3. Load dataset from JSONL file
    4. Filter out empty texts
    5. Apply preprocessing pipeline with multiprocessing
    6. Save processed data to output file
    7. Log completion statistics
    """
    # Parse command line arguments
    args = get_args()
    disease = args.disease
    outdir = args.outdir
    outfile = args.outfile
    infile = args.infile
    indir = os.path.join(args.indir, disease)
    
    # Set up logging for monitoring preprocessing progress
    logger = setup_logger("preprocessor", os.path.join(outdir, "preprocessor.log"))
    logger.info(f"disease:{disease}\nindir:{indir}\noutdir:{outdir}\noutfile:{outfile}\n")

    # Load dataset from JSONL file using HuggingFace datasets
    dataset_dict = load_dataset('json', data_files=os.path.join(indir, infile))
    dataset = dataset_dict['train']
    
    # Filter out examples with empty text fields to avoid processing errors
    filtered_dataset = dataset.filter(lambda example: example['text'].strip() != '')
    
    # Apply preprocessing pipeline with parallel processing for efficiency
    # Using 16 processes to speed up text processing on large datasets
    processed_dataset = filtered_dataset.map(preprocess_example, batched=True, num_proc=16)
    
    # Save processed dataset to output JSONL file
    save_to_jsonl(processed_dataset, os.path.join(outdir, outfile), mode='a')
    
    # Log final dataset statistics for verification
    logger.info(processed_dataset)