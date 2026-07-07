import os
import json

def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))

        start += chunk_size - overlap # move with overlap

    return chunks

def process_files_in_folder(folder_path, output_file):
    """Read all files in a folder, chunk their content and save the results."""
    all_chunks = {}
    chunk_index = 0 # Start with a global index for all chunks

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Chunk the content
                chunks = chunk_text(content)

                # Assign simple integer indices to chunks
                for chunk in chunks:
                    all_chunks[chunk_index] = chunk
                    chunk_index += 1

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    # Save all chunks to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as out_file:
            json.dump(all_chunks, out_file, ensure_ascii=False, indent=4)
        print(f"Chunks saved to {output_file}")
    except Exception as e:
        print(f"Error saving chunks to file: {e}")


def main():
    # Define the folder path and output file
    folder_path = os.path.join(os.path.dirname(__file__), 'data')
    output_file = os.path.join(os.path.dirname(__file__), 'chunked_data.json')

    # Process all files in the folder
    process_files_in_folder(folder_path, output_file)

if __name__ == "__main__":
    main()
