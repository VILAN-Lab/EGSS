ner_file = './train_ner50.txt'
bio_file = './train_bio50.txt'
tgt_file = './train_tgt50.txt'

f1 = open('.squad/train_entity50.txt', 'a', encoding='utf-8')
with open(ner_file, 'r', encoding='utf-8') as ner, open(bio_file, 'r', encoding='utf-8') as bio:
    for ne, bi in zip(ner, bio):
        n = ne.split()
        b = bi.split()

        list1 = []
        list_ner = []
        list_tgt = []
        m = " "
        for i in range(len(b)):
            if b[i] != "O":
                list1.append(b[i])
                list_ner.append(n[i])
                m = n[i]+" "
                # print("list_ner:", m)
        # f1.writelines(list_ner + "\n")
        c = " "
        for i in list_ner:
            print("i:", i)
            c = c + " " + i
        f1.writelines(c + "\n")
        # print("list1:", list1)
        # print("c:", c)
        # print("list_ner:", list_ner)
        # print(ne)
f1.close()




