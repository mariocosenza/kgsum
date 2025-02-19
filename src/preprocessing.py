from os import listdir

import spacy
import pandas as pd
from deep_translator import GoogleTranslator

nlp = spacy.load("en_core_web_sm")


def merge_dataset() -> pd.DataFrame:
    local_frames = []
    for file in listdir('../data/raw/local'):
        if 'local_feature_set' in file:
            local_frames.append(pd.read_json(f'../data/raw/local/{file}'))

    remote_frames = []
    for file in listdir('../data/raw/remote'):
        if 'remote_feature_set' in file:
            remote_frames.append(pd.read_json(f'../data/raw/remote/{file}'))

    # Concatenate all frames from local and remote directories.
    # If either list is empty, use an empty DataFrame.
    df_local = pd.concat(local_frames, ignore_index=True) if local_frames else pd.DataFrame()
    df_remote = pd.concat(remote_frames, ignore_index=True) if remote_frames else pd.DataFrame()

    # Combine the two DataFrames.
    merged_df = pd.concat([df_local, df_remote], ignore_index=True)

    # Drop duplicates based on the "id" column.
    # When duplicates occur, the row from the remote (later in the concatenation) is kept.
    merged_df = merged_df.drop_duplicates(subset='id', keep='last')

    return merged_df


def merge_void_dataset():
    local_frames = []
    for file in listdir('../data/raw/local'):
        if 'local_void_feature_set' in file:
            local_frames.append(pd.read_json(f'../data/raw/local/{file}'))

    remote_frames = []
    for file in listdir('../data/raw/remote'):
        if 'remote_void_feature_set' in file:
            remote_frames.append(pd.read_json(f'../data/raw/remote/{file}'))

    df_local = pd.concat(local_frames, ignore_index=True) if local_frames else pd.DataFrame()
    df_remote = pd.concat(remote_frames, ignore_index=True) if remote_frames else pd.DataFrame()

    merged_df = pd.concat([df_local, df_remote], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset='id', keep='last')

    return merged_df

def _translate_text(word_list: [], translator) -> str:
    if len(word_list) <= 5000:
        text = ''
        for word in word_list:
            text += word + ' '
            return translator.translate(text)
    else:
        size = 0
        sliced_words = ''
        temp_sliced_words = ''
        for word in word_list:
            if len(word) <= 5000 - size:
                temp_sliced_words = temp_sliced_words + + word + ' '
                size += len(word)
            else:
                size = 0
                sliced_words = sliced_words + ' ' +  translator.translate(temp_sliced_words) + ' ' +  translator.translate(word)
                temp_sliced_words = ''
        return translator.translate(sliced_words)

def preprocess_lab_lcn_lnp(input_frame: pd.DataFrame):
    translator = GoogleTranslator(source='auto', target='en')
    list_lab = list()
    list_lnp = list()
    list_lcn = list()
    for index, row in  input_frame.iterrows(): #todo Keep in mind NoneType and data Imputation
        translated_lcn_row = _translate_text(row['lcn'], translator)
        translated_lab_row = _translate_text(row['lab'], translator)
        translated_lpn_row = _translate_text(row['lpn'], translator)
        cleaned_lcn_row = [word.text for word in nlp(translated_lcn_row) if not word.like_url]
        cleaned_lab_row = [word.text for word in nlp(translated_lab_row) if not word.like_url]
        cleaned_lpn_row = [word.text for word in nlp(translated_lpn_row) if not word.like_url]
        cleaned_lab_row = nlp.pipe(cleaned_lab_row, n_process=-1)
        cleaned_lcn_row = nlp.pipe(cleaned_lcn_row, n_process=-1)
        cleaned_lpn_row = nlp.pipe(cleaned_lpn_row, n_process=-1)
        list_lab.append(cleaned_lab_row)
        list_lnp.append(cleaned_lpn_row)
        list_lcn.append(cleaned_lcn_row)

    pd.DataFrame({
        "id": input_frame["id"],
        "category": input_frame["category"],
        "lcn": list_lcn,
        "lab": list_lab,
        "lnp": list_lnp
    }).to_json('../data/processed/lab_lcn_lnp.json')




def preprocess_void(input_frame):
    processed_frame = pd.DataFrame()
    translator = GoogleTranslator(source='auto', target='en')
    for row in input_frame:
        translated_dsc_row = _translate_text(row['dsc'], translator)
        cleaned_dsc_row = [word.text for word in nlp(translated_dsc_row) if not word.like_url]
        cleaned_dsc_row = nlp.pipe(cleaned_dsc_row, n_process=-1)



def preprocess_voc_tags(input_frame):
    for row in input_frame:
        cleaned_voc_row = nlp.pipe(row['tags'], n_process=-1)


def preprocess_voc_curi_puri_tld(input_frame):
    processed_frame = pd.DataFrame()


print(merge_dataset())