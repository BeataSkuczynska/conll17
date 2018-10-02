import codecs


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def save_conll(formatted_sents, output_path):
    with codecs.open(output_path, "w", "utf-8") as f:
        for sentence in formatted_sents:
            for word in sentence:
                if type(word) is list:
                    f.write("\t".join(word) + "\n")
                else:
                    f.write(word + "\n")
            f.write("\n")
