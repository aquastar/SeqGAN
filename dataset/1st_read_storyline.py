import itertools
import cPickle as pk

file = open("storyline_train")

storyline = {}
while 1:
    lines = file.readlines(100000)
    if not lines:
        break
    for line in lines:
        id_story = line.split(':')
        sid = id_story[0]
        story = id_story[1].strip()
        entity = story.split(',')

        multiopt = []
        multiopt_id = []
        uniopt = []
        uniopt_id = []

        for eid, en in enumerate(entity):
            if '|' in en:
                multiopt_id.append(eid)
                multiopt.append(en.split('|'))
            else:
                uniopt_id.append(eid)
                uniopt.append(en)

        for en in itertools.product(*multiopt):
            new_story = list(en)
            for uid, uen in zip(uniopt_id, uniopt):
                new_story.insert(uid, uen)

            print new_story

            if sid in storyline:
                storyline[sid].append(new_story)
            else:
                storyline[sid] = [new_story]

pk.dump(storyline, open('storyline.pk', 'wb'))
