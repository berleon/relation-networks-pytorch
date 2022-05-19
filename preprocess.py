import os
import sys
import json
import pickle
from typing import Optional

import nltk
import tqdm
from torchvision import transforms
from PIL import Image


def get_question_filename(root, split):
    return os.path.join(root, "questions", f"CLEVR_{split}_questions.json")


def get_result_file(root, split):
    return os.path.join(root, f"{split}.pkl")


def process_question(
    root,
    split,
    word_dic=None,
    answer_dic=None,
    question_file: Optional[str] = None,
    result_file: Optional[str] = None,
):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    if question_file is None:
        question_file = get_question_filename(root, split)

    if result_file is None:
        result_file = get_result_file(root, split)

    with open(question_file) as f:
        data = json.load(f)

    result = []
    word_index = 1
    answer_index = 0

    for question in tqdm.tqdm(data["questions"]):
        words = nltk.word_tokenize(question["question"])
        question_token = []

        for word in words:
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = question["answer"]

        try:
            answer = answer_dic[answer_word]

        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1

        result.append(
            (
                question["image_filename"],
                question_token,
                answer,
                question["question_family_index"],
            )
        )

    with open(result_file, "wb") as f:
        pickle.dump(result, f)

    return word_dic, answer_dic


resize = transforms.Resize([128, 128])


def process_image(path, output_dir):
    images = os.listdir(path)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for imgfile in tqdm.tqdm(images):
        img = Image.open(os.path.join(path, imgfile)).convert("RGB")
        img = resize(img)
        img.save(os.path.join(output_dir, imgfile))


def main(root, only_questions=False):

    word_dic, answer_dic = process_question(root, "train")
    process_question(root, "val", word_dic, answer_dic)

    with open(os.path.join(root, "dic.pkl"), "wb") as f:
        pickle.dump({"word_dic": word_dic, "answer_dic": answer_dic}, f)

    if only_questions:
        return

    process_image(
        os.path.join(root, "images/train"),
        os.path.join(root, "images/train_preprocessed"),
    )
    process_image(
        os.path.join(root, "images/val"),
        os.path.join(root, "images/val_preprocessed"),
    )


if __name__ == "__main__":
    main(sys.argv[1])
