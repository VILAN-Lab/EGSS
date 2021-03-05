
from collections import Counter
from contextlib import ExitStack

tgt_f = "./train_tgt50.txt"
enti_f = "./train_entity50.txt"
ans_pos_f = "./train_ans_pos50.txt"
ans_file = "./train_answer50.txt"
src_file = "./train_src50.txt"

k = 0
i = 0
m = 0
h_tj = ["infinitely","around half","second-most","millions","half ","twice","second-most populous","One-quarter","rarely","months","scourge","ten-horsepower","512-bit"," largest","increasingly","often","in recent decades",\
        "dozens","homologous","no damage","voluminous","most common","simple majority","fundraising","confirmation"," rate ","several","minimal ","12-screen"," few ","sevenfold","24-year","minute","biennial","thousands","annually",""\
        "limited","a few","a year","none","nothing","half-penny","sync","two-thirds","eight-year","third-most","disaster","25-foot","30-second",
        ]
where_tj = ["highlands","us/canada","special university classes","premises of the hospital","underground","oceans","southeast","the northeast","courtyard","sidelines",\
        "magnetic fields","the clergy","north and east","easterly","the north ","eastern","the castle","the stems","granules","museum","lehramstudien","thylakoid space","divergent boundaries",\
        "maria fold","ilkhanate","contries","highest","the middle","satellites","a satellite","leaves","principal","the western",\
        "private sector","roof","angiosperms","c4 plants","battle of hastings","tours","foot ","outdoors","mountainous areas","kingdom ","bays","dams","branched"]


what1_tj = ["67.9","90 ","4.7","125p","0.519","3.62","11.4","6.7","0.519","7,200","0.52","1.3",\
        "80.4","55,000","1,655,114","2,400","3,792621"
        ]
who_misc = ["muslims","naimans","seljuk","turks","han chinese","anglo-saxons","americans"
        ]
which_misc = ["renaissance","portuguese","israelis","muslim","ethiopian empire","northern chinese","sourth chinese","coptic","lgbt","japanese","super bowl xlvii","harrods",\
        "mexican","pre-columbian","british empire","swiss-austrian"
        ]
when_misc = ["roman empire","super bowl xxviii","christmas day"
        ]
how_percent = ["12 %","500 % ","30 %","14 %","50 %","80 %","2.4 %","30 %","24 %","42 %"]

f1 = open('./train_ent_qtype50.txt', 'a', encoding='utf-8')

with ExitStack() as stack:
    files = [stack.enter_context(open(fname)) for fname in \
             [tgt_f, enti_f, ans_pos_f, ans_file, src_file]]
    for t, e, ap, a, s in zip(files[0], files[1], files[2], files[3], files[4]):
        a = a.strip()
        pp = ap.split()
        enl = e.split()
        dic = Counter(enl)

        if "due to" in a:
            c = "why"
            f1.writelines(c + "\n")
        elif "because" in a:
            c = "why"
            f1.writelines(c + "\n")
#         enl = en.split()
#         tl = t.split()
        else:
            if "organization" in enl:
                c = "what"
                f1.writelines(c + "\n")

            elif "ordinal" in enl:
                c = "what"
                f1.writelines(c + "\n")

            elif "location" in enl:
                c = "where"
                f1.writelines("what" + " " + c + "\n")

            elif "percent" in enl:
                c = "what"
                f1.writelines(c + " " + "how" + "\n")

            elif "money" in enl:
                c = "how"
                f1.writelines("what" + " " + c + "\n")


            elif "date" in enl:
                c = "when"
                f1.writelines(c + "\n")

            elif "time" in enl:
                c = "when"
                f1.writelines(c + "\n")
            elif len(dic.keys()) == 1:
                for k in dic:
                    if  k == "person":
                        f1.writelines("who" + "\n")
                    elif k == "o":

                        if "NN" in pp:
                            f1.writelines("what" + "\n")
                        elif "NNS" in pp:
                            f1.writelines("what" + "\n")
                        elif "NNP" in pp:
                            f1.writelines("what" + "\n")
                        elif "NNPS" in pp:
                            f1.writelines("what" + "\n")
                        elif "JJ" in pp:
                            f1.writelines("what" + "\n")

                        else:
                            ls = []
                            for i in h_tj:
                                if i in a:
                                    ls.append("how")
                            for j in where_tj:
                                if j in a:
                                    ls.append("where")
                            ls.append("other")
                            f1.writelines(ls[0] + "\n")

                    elif k == "number":
                        la = []
                        for i in what1_tj:
                            if i in a:
                                la.append("what")

                        la.append("how")
                        f1.writelines(la[0] + "\n")

                    elif k == "misc":
                        la1 = []
                        for i in who_misc:
                            if i in a:
                                la1.append("who")

                        for i in which_misc:
                            if i in a:
                                la1.append("which")

                        for i in when_misc:
                            if i in a:
                                la1.append("when")

                        la1.append("what")
                        f1.writelines(la1[0] + "\n")

                    else:
                        f1.writelines("other" + "\n")
            elif "person" in enl:
                c = "who"
                f1.writelines(c + "\n")
            elif "number" in enl:
                c = "how"
                f1.writelines(c + " " + "what " + "\n")
            elif "misc" in enl:
                c = "what"
                f1.writelines(c + "\n")
            else:
                c = "other"
                f1.writelines(c + "\n")
f1.close()
