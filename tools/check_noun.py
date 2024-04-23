import json
import argparse
import os
from multiprocessing import Pool

# from joblib import Parallel, delayed
from tqdm import tqdm
import random
from functools import partial
import spacy
import time
import en_core_web_trf

parser = argparse.ArgumentParser(description="Read json")
parser.add_argument('file', type=str)
parser.add_argument('-m', '--multi', type=str, default=False)
parser.add_argument('-p', '--process', type=int, default=4)
args = parser.parse_args()

nlp = en_core_web_trf.load()


def multi_check_jsonl(f):
    num = 0
    odvg_annos = []
    num_entity = 0
    (
        CARDINAL,
        DATE,
        EVENT,
        FAC,
        GPE,
        LANGUAGE,
        LAW,
        LOC,
        MONEY,
        NORP,
        ORDINAL,
        ORG,
        PERCENT,
        PERSON,
        PRODUCT,
        QUANTITY,
        TIME,
        WORK_OF_ART,
    ) = [[] for _ in range(18)]
    start_time = time.time()
    num_jsonl = len(f.readlines())
    f.seek(0)
    if args.multi:
        with Pool(args.process) as pool:
            iters = pool.imap(func=check_noun, iterable=f)
            for iter in tqdm(iters, total=num_jsonl):
                odvg_annos.extend(iter[0])
                num += iter[1]
                num_entity += iter[2]
                CARDINAL.extend(iter[3])
                DATE.extend(iter[4])
                EVENT.extend(iter[5])
                FAC.extend(iter[6])
                GPE.extend(iter[7])
                LANGUAGE.extend(iter[8])
                LAW.extend(iter[9])
                LOC.extend(iter[10])
                MONEY.extend(iter[11])
                NORP.extend(iter[12])
                ORDINAL.extend(iter[13])
                ORG.extend(iter[14])
                PERCENT.extend(iter[15])
                PERSON.extend(iter[16])
                PRODUCT.extend(iter[17])
                QUANTITY.extend(iter[18])
                TIME.extend(iter[19])
                WORK_OF_ART.extend(iter[20])
                # print(iter)
    else:
        for j in tqdm(f, total=num_jsonl):
            iter = check_noun(j)
            odvg_annos.extend(iter[0])
            num += iter[1]
            num_entity += iter[2]
            CARDINAL.extend(iter[3])
            DATE.extend(iter[4])
            EVENT.extend(iter[5])
            FAC.extend(iter[6])
            GPE.extend(iter[7])
            LANGUAGE.extend(iter[8])
            LAW.extend(iter[9])
            LOC.extend(iter[10])
            MONEY.extend(iter[11])
            NORP.extend(iter[12])
            ORDINAL.extend(iter[13])
            ORG.extend(iter[14])
            PERCENT.extend(iter[15])
            PERSON.extend(iter[16])
            PRODUCT.extend(iter[17])
            QUANTITY.extend(iter[18])
            TIME.extend(iter[19])
            WORK_OF_ART.extend(iter[20])
    print('\nnum_entity:', num_entity)
    with open('grit_v1_min3_noun_ban.log', 'w') as f:
        f.write(f'CARDINAL:{len(CARDINAL)}\n {CARDINAL}\n')
        f.write(f'DATE:{len(DATE)}\n {DATE}\n')
        f.write(f'EVENT:{len(EVENT)}\n {EVENT}\n')
        f.write(f'FAC:{len(FAC)}\n {FAC}\n')
        f.write(f'GPE:{len(GPE)}\n {GPE}\n')
        f.write(f'LANGUAGE:{len(LANGUAGE)}\n {LANGUAGE}\n')
        f.write(f'LAW:{len(LAW)}\n {LAW}\n')
        f.write(f'LOC:{len(LOC)}\n {LOC}\n')
        f.write(f'MONEY:{len(MONEY)}\n {MONEY}\n')
        f.write(f'NORP:{len(NORP)}\n {NORP}\n')
        f.write(f'ORDINAL:{len(ORDINAL)}\n {ORDINAL}\n')
        f.write(f'ORG:{len(ORG)}\n {ORG}\n')
        f.write(f'PERCENT:{len(PERCENT)}\n {PERCENT}\n')
        f.write(f'PERSON:{len(PERSON)}\n {PERSON}\n')
        f.write(f'PRODUCT:{len(PRODUCT)}\n {PRODUCT}\n')
        f.write(f'QUANTITY:{len(QUANTITY)}\n {QUANTITY}\n')
        f.write(f'TIME:{len(TIME)}\n {TIME}\n')
        f.write(f'WORK_OF_ART:{len(WORK_OF_ART)}\n {WORK_OF_ART}\n')
    end_time = time.time()
    print("程序运行时间：", end_time - start_time, "秒")


def check_noun(line):
    num = 0
    odvg_annos = []
    num_entity = 0
    (
        CARDINAL,
        DATE,
        EVENT,
        FAC,
        GPE,
        LANGUAGE,
        LAW,
        LOC,
        MONEY,
        NORP,
        ORDINAL,
        ORG,
        PERCENT,
        PERSON,
        PRODUCT,
        QUANTITY,
        TIME,
        WORK_OF_ART,
    ) = [[] for _ in range(18)]

    j = json.loads(line)

    ## do nothing
    # odvg_annos.append(j)
    # num += 1
    # if num == 200:
    #     break

    ## check noun
    sentence = j['grounding']['caption']
    # print('nlp processing now...')
    doc = nlp(sentence)
    # print('nlp processing end')
    # return doc[0].text
    # ban_entity_labels = [
    #     "CARDINAL",
    #     "DATE",
    #     "EVENT",
    #     "GPE",
    #     "LANGUAGE",
    #     "LAW",
    #     "MONEY",
    #     "NORP",
    #     "ORDINAL",
    #     "ORG",
    #     "PERCENT",
    #     "PERSON",
    #     "PRODUCT",
    #     "QUANTITY",
    #     "TIME",
    # ]
    all_labels_trf = [
        "CARDINAL",
        "DATE",
        "EVENT",
        "FAC",
        "GPE",
        "LANGUAGE",
        "LAW",
        "LOC",
        "MONEY",
        "NORP",
        "ORDINAL",
        "ORG",
        "PERCENT",
        "PERSON",
        "PRODUCT",
        "QUANTITY",
        "TIME",
        "WORK_OF_ART",
    ]
    # entities = [[ent.text, ent.label_] for ent in doc.ents if ent.label_ in ban_entity_labels]
    # entities_text = [ent[0] for ent in entities]
    entities = {ent.text: ent.label_ for ent in doc.ents if ent.label_ in all_labels_trf}
    entities_text = entities.keys()
    # for anno in annos:
    #     print(anno['phrase'], end=', ')
    regions = []
    for anno in j['grounding']['regions']:
        if anno['phrase'] in entities_text:
            if entities[anno['phrase']] == 'CARDINAL':
                CARDINAL.append(anno['phrase'])
            elif entities[anno['phrase']] == 'DATE':
                DATE.append(anno['phrase'])
            elif entities[anno['phrase']] == 'EVENT':
                EVENT.append(anno['phrase'])
            elif entities[anno['phrase']] == 'FAC':
                FAC.append(anno['phrase'])
            elif entities[anno['phrase']] == 'GPE':
                GPE.append(anno['phrase'])
            elif entities[anno['phrase']] == 'LANGUAGE':
                LANGUAGE.append(anno['phrase'])
            elif entities[anno['phrase']] == 'LAW':
                LAW.append(anno['phrase'])
            elif entities[anno['phrase']] == 'LOC':
                LOC.append(anno['phrase'])
            elif entities[anno['phrase']] == 'MONEY':
                MONEY.append(anno['phrase'])
            elif entities[anno['phrase']] == 'NORP':
                NORP.append(anno['phrase'])
            elif entities[anno['phrase']] == 'ORDINAL':
                ORDINAL.append(anno['phrase'])
            elif entities[anno['phrase']] == 'ORG':
                ORG.append(anno['phrase'])
            elif entities[anno['phrase']] == 'PERCENT':
                PERCENT.append(anno['phrase'])
            elif entities[anno['phrase']] == 'PERSON':
                PERSON.append(anno['phrase'])
            elif entities[anno['phrase']] == 'PRODUCT':
                PRODUCT.append(anno['phrase'])
            elif entities[anno['phrase']] == 'QUANTITY':
                QUANTITY.append(anno['phrase'])
            elif entities[anno['phrase']] == 'TIME':
                TIME.append(anno['phrase'])
            elif entities[anno['phrase']] == 'WORK_OF_ART':
                WORK_OF_ART.append(anno['phrase'])
            # print('sentence:', sentence)
            # print('entities:', entities)
            # print('entity phrase:', end=' ')
            # print(anno['phrase'])
            num_entity += 1
            continue
        regions.append(anno)
    if regions != []:
        odvg = j
        odvg['grounding']['regions'] = regions
        odvg_annos.append(odvg)
        num += 1
        # if num == 200:
        #     break
    return [
        odvg_annos,
        num,
        num_entity,
        CARDINAL,
        DATE,
        EVENT,
        FAC,
        GPE,
        LANGUAGE,
        LAW,
        LOC,
        MONEY,
        NORP,
        ORDINAL,
        ORG,
        PERCENT,
        PERSON,
        PRODUCT,
        QUANTITY,
        TIME,
        WORK_OF_ART,
    ]
    # print(odvg_annos)
    # json_name = 'grit_00000_v2_fromv1.jsonl'
    # with jsonlines.open(json_name, mode="w") as fwriter:
    #     fwriter.write_all(odvg_annos)


def check_noun_(line):
    num = 0
    odvg_annos = []
    num_entity = 0
    (
        CARDINAL,
        DATE,
        EVENT,
        FAC,
        GPE,
        LANGUAGE,
        LAW,
        LOC,
        MONEY,
        NORP,
        ORDINAL,
        ORG,
        PERCENT,
        PERSON,
        PRODUCT,
        QUANTITY,
        TIME,
        WORK_OF_ART,
    ) = [[] for _ in range(18)]

    j = json.loads(line)

    ## do nothing
    # odvg_annos.append(j)
    # num += 1
    # if num == 200:
    #     break

    ## check noun
    sentence = j['grounding']['caption']
    # print('nlp processing now...')
    doc = nlp(sentence)
    # print('nlp processing end')
    # return doc[0].text
    # ban_entity_labels = [
    #     "CARDINAL",
    #     "DATE",
    #     "EVENT",
    #     "GPE",
    #     "LANGUAGE",
    #     "LAW",
    #     "MONEY",
    #     "NORP",
    #     "ORDINAL",
    #     "ORG",
    #     "PERCENT",
    #     "PERSON",
    #     "PRODUCT",
    #     "QUANTITY",
    #     "TIME",
    # ]
    all_labels_trf = [
        "CARDINAL",
        "DATE",
        "EVENT",
        "FAC",
        "GPE",
        "LANGUAGE",
        "LAW",
        "LOC",
        "MONEY",
        "NORP",
        "ORDINAL",
        "ORG",
        "PERCENT",
        "PERSON",
        "PRODUCT",
        "QUANTITY",
        "TIME",
        "WORK_OF_ART",
    ]
    # entities = [[ent.text, ent.label_] for ent in doc.ents if ent.label_ in ban_entity_labels]
    # entities_text = [ent[0] for ent in entities]
    entities = {ent.text: ent.label_ for ent in doc.ents if ent.label_ in all_labels_trf}
    entities_text = entities.keys()
    # for anno in annos:
    #     print(anno['phrase'], end=', ')
    regions = []
    for anno in j['grounding']['regions']:
        if anno['phrase'] in entities_text:
            if entities[anno['phrase']] == 'CARDINAL':
                CARDINAL.append(anno['phrase'])
            elif entities[anno['phrase']] == 'DATE':
                DATE.append(anno['phrase'])
            elif entities[anno['phrase']] == 'EVENT':
                EVENT.append(anno['phrase'])
            elif entities[anno['phrase']] == 'FAC':
                FAC.append(anno['phrase'])
            elif entities[anno['phrase']] == 'GPE':
                GPE.append(anno['phrase'])
            elif entities[anno['phrase']] == 'LANGUAGE':
                LANGUAGE.append(anno['phrase'])
            elif entities[anno['phrase']] == 'LAW':
                LAW.append(anno['phrase'])
            elif entities[anno['phrase']] == 'LOC':
                LOC.append(anno['phrase'])
            elif entities[anno['phrase']] == 'MONEY':
                MONEY.append(anno['phrase'])
            elif entities[anno['phrase']] == 'NORP':
                NORP.append(anno['phrase'])
            elif entities[anno['phrase']] == 'ORDINAL':
                ORDINAL.append(anno['phrase'])
            elif entities[anno['phrase']] == 'ORG':
                ORG.append(anno['phrase'])
            elif entities[anno['phrase']] == 'PERCENT':
                PERCENT.append(anno['phrase'])
            elif entities[anno['phrase']] == 'PERSON':
                PERSON.append(anno['phrase'])
            elif entities[anno['phrase']] == 'PRODUCT':
                PRODUCT.append(anno['phrase'])
            elif entities[anno['phrase']] == 'QUANTITY':
                QUANTITY.append(anno['phrase'])
            elif entities[anno['phrase']] == 'TIME':
                TIME.append(anno['phrase'])
            elif entities[anno['phrase']] == 'WORK_OF_ART':
                WORK_OF_ART.append(anno['phrase'])
            # print('sentence:', sentence)
            # print('entities:', entities)
            # print('entity phrase:', end=' ')
            # print(anno['phrase'])
            num_entity += 1
            continue
        regions.append(anno)
    if regions != []:
        odvg = j
        odvg['grounding']['regions'] = regions
        odvg_annos.append(odvg)
        num += 1
        # if num == 200:
        #     break
    return [
        odvg_annos,
        num,
        num_entity,
        CARDINAL,
        DATE,
        EVENT,
        FAC,
        GPE,
        LANGUAGE,
        LAW,
        LOC,
        MONEY,
        NORP,
        ORDINAL,
        ORG,
        PERCENT,
        PERSON,
        PRODUCT,
        QUANTITY,
        TIME,
        WORK_OF_ART,
    ]
    # print(odvg_annos)
    # json_name = 'grit_00000_v2_fromv1.jsonl'
    # with jsonlines.open(json_name, mode="w") as fwriter:
    #     fwriter.write_all(odvg_annos)


def main(args):
    with open(args.file, 'r') as f:
        print('loading...')
        # #check jsonl noun

        multi_check_jsonl(f)

        # num_jsonl = len(f.readlines())
        # f.seek(0)
        # for line in tqdm(f, total=num_jsonl):
        #     iter = check_noun(line)


if __name__ == '__main__':
    main(args)
