golden_file = "./train_answer50bio.txt"
gen_file = "./train_other-ans-style.txt"

f =open("./train_ans-style-oth.txt", 'a', encoding="utf-8")
i=0
k=0
with open(golden_file, 'r', encoding="utf-8") as gf, open(gen_file, 'r', encoding='utf-8') as pf:
    for g, p in zip(gf, pf):
        g = g.strip()
        p = p.strip()
        a = g + " " + "</s>" + " " + p
        k = k+1
        f.write(a+"\r\n")

f.close()