#!/bin/bash

# Initialize default values
input_file=""
output_file=""
tokenizer_name=""
tokenizer_type="hf"
start_id=-1
start_column=false
legacy_trie=false

# ./scripts/trie/build_perm_trie -i nq_unique_pseudo_queries.txt -o nq_unique_pseudo_queries_pseudo_query_500K_strict_special_trie_new -t /data_ecstorage/MINDER/tmp/vocabs/nq_pseudo_query_500K_strict_special.yaml -T tokenmonster -s 0


# Function to display help message
show_help() {
cat << EOF
Usage: ${0##*/} [options]
This script builds a trie data structure from a raw text file.
Example: ./build_trie -i input.txt -o trie -t microsoft/phi-2

    -i, FILE      Specify the input file with raw text and optionally the prefix ids (required).
    -o, FILE      Specify the output trie file (required).
    -t, NAME      Specify the tokenizer name or path (required).
    -T, TYPE      Specify the tokenizer type. Options are 'hf' (HuggingFace),
                  'tokenizers', or 'tokenmonster' (default: 'hf').
    -s, ID        Specify the start ID. A prefix added to every sentence in the trie.
                  Use -1 for no prefix (default: -1).
    -c,           Specify if the input file has start IDs in the first column and
                  sentences in the second column (default: false).
    -l,           Use legacy trie binaries, not recommended (default: false)
    -h,           Display this help and exit.
EOF
}

# Validate tokenizer type
validate_tokenizer_type() {
  case $1 in
    hf|tokenizers|tokenmonster|deberta_spm)
      return 0
      ;;
    *)
      echo "Invalid tokenizer type: $1. Valid options are 'hf', 'tokenizers', 'deberta_spm', or 'tokenmonster'."
      exit 1
      ;;
  esac
}

# Parse command-line options
while getopts ":i:o:t:T:s:clh" opt; do
  case ${opt} in
    i )
      input_file=$OPTARG
      ;;
    o )
      output_file=$OPTARG
      ;;
    t )
      tokenizer_name=$OPTARG
      ;;
    T )
      tokenizer_type=$OPTARG
      validate_tokenizer_type "$tokenizer_type"
      ;;
    s )
      start_id=$OPTARG
      ;;
    c )
      start_column=true
      ;;
    l )
      legacy_trie=true
      ;;
    h )
      show_help
      exit 0
      ;;
    \? )
      echo "Invalid Option: -$OPTARG" 1>&2
      exit 1
      ;;
    : )
      echo "Invalid Option: -$OPTARG requires an argument" 1>&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

# Check for required arguments
if [ -z "$input_file" ] || [ -z "$output_file" ] || [ -z "$tokenizer_name" ]; then
  echo "Error: Input file, output file, and tokenizer name are required."
  show_help
  exit 1
fi

# Rest of the script that uses the variables set by the command-line options
echo "Specified Arguments"
printf '=%.0s' {1..20}; printf '\n'
echo "Input File: $input_file"
echo "Output File: $output_file"
echo "Tokenizer Name: $tokenizer_name"
echo "Tokenizer Type: $tokenizer_type"
echo "Start ID: $start_id"
echo "Start Column: $start_column"
echo "Legacy Trie: $legacy_trie"
printf '=%.0s' {1..20}; printf '\n'

hash=`echo $tokenizer_name $tokenizer_type | md5sum | cut -f1 -d" "`
uid=${hash: -5}

cmd="python trie/tokenize_and_convert_to_ids.py \
        --input-file ${input_file} \
        --output-file ${output_file}.tokenized${uid} \
        --tokenizer-name ${tokenizer_name} \
        --tokenizer-type ${tokenizer_type} \
        --start-id ${start_id}"

if [[ ${start_column} = "true" ]];then
    cmd="$cmd --start-column"
fi
echo "Executing $cmd"
eval $cmd
echo "Completed tokenization"

# Add 4 permuations for MSMARCO, 8 for NQ -- this is based on the number of docids per passage available for the dataset
python scripts/trie/add_permutations.py --tokenized_sentences_file ${output_file}.tokenized${uid} --output_file ${output_file}.tokenized${uid}.perm8 --max_permutations 8
