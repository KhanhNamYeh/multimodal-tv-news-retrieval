import os
import json


def merge_transcript_and_caption(transcript_file, time_extract_file):
    """
    Merge transcript (text) and time_extract (caption) files.
    Returns None if time_extract file does not exist or has no captions.
    Sort by start time; if equal, caption comes before text.
    Returns concatenated string of all content.
    """
    # Skip if time_extract file doesn't exist
    if not time_extract_file or not os.path.exists(time_extract_file):
        return None

    with open(time_extract_file, 'r', encoding='utf-8') as f:
        time_extract_data = json.load(f)

    # Skip if no caption fields found
    if not any('caption' in item for item in time_extract_data):
        return None

    entries = []

    # Load transcript entries
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)

    for item in transcript_data:
        entries.append({
            'start': item['start'],
            'type': 'text',
            'content': item['text']
        })

    # Add caption entries from time_extract
    for item in time_extract_data:
        if 'caption' in item:
            entries.append({
                'start': item['start'],
                'type': 'caption',
                'content': item['caption']
            })

    # Sort by start time; caption before text if start is equal
    entries.sort(key=lambda x: (x['start'], 0 if x['type'] == 'caption' else 1))

    # Concatenate all content into one string
    return ' '.join(entry['content'] for entry in entries)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    transcripts_dir = os.path.join(base_dir, 'dataset', 'transcripts')
    time_extract_dir = os.path.join(base_dir, 'dataset', 'time_extract')
    concat_dir = os.path.join(base_dir, 'dataset', 'concat')

    os.makedirs(concat_dir, exist_ok=True)

    if not os.path.isdir(transcripts_dir):
        print(f"ERROR: transcripts folder not found: {transcripts_dir}")
        return

    folder_names = sorted(
        name for name in os.listdir(transcripts_dir)
        if os.path.isdir(os.path.join(transcripts_dir, name))
    )

    for folder_name in folder_names:
        folder_path = os.path.join(transcripts_dir, folder_name)
        output_entries = []

        json_files = sorted(
            f for f in os.listdir(folder_path)
            if f.endswith('.json')
        )

        for filename in json_files:
            file_id = filename.replace('.json', '')  # e.g. "000"
            transcript_file = os.path.join(folder_path, filename)

            # time_extract path: time_extract/K01_V001/K01_V001_000.json
            time_extract_filename = f"{folder_name}_{file_id}.json"
            time_extract_file = os.path.join(time_extract_dir, folder_name, time_extract_filename)

            text = merge_transcript_and_caption(transcript_file, time_extract_file)

            # Skip if time_extract doesn't exist or has no captions
            if text is None:
                continue

            output_entries.append({
                'id': file_id,
                'text': text
            })

        if output_entries:
            output_file = os.path.join(concat_dir, f"{folder_name}.json")
            if os.path.exists(output_file):
                print(f"Skipped: concat/{folder_name}.json  (already exists)")
                continue
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_entries, f, ensure_ascii=False, indent=2)
            print(f"Created: concat/{folder_name}.json  ({len(output_entries)} segments)")

    print("\nDone.")


if __name__ == '__main__':
    main()
