import json
import os
import librosa
import warnings

warnings.filterwarnings('ignore')
DATASET_PATH = "dataset"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050
DURATION = 0.5  # measured in second
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = num_samples_per_segment / hop_length

    # loop through all the genres
    for i, (dirpath, dirnames, filesname) in enumerate(os.walk(dataset_path)):

        # ensure that we're not at the root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split("/")  # class/chainsaw => ["class", "chainsaw"]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nprocessing {}".format(semantic_label))

            # process files for a specific class
            for f in filesname:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                # if signal[0] != 22050:
                #     signal = librosa.util.fix_length(signal, 22050)
                # process segment extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s  # s=0 -> 0
                    finsih_sample = start_sample + num_samples_per_segment  # s=0 -> num_samples_per_segment

                    mfcc = librosa.feature.mfcc(signal, sr=sr, n_fft=n_fft, n_mfcc=num_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T
                    # print(len(mfcc))
                    
                    # store only mfcc feature with expected number of vectors
                    # if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i-1)
                    print("{}, segment:{}".format(file_path, s+1))
                    

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)