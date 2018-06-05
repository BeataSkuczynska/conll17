def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def save_conll(formatted_sents, output_path):
    with open(output_path, "w") as f:
        for sentence in formatted_sents:
            for word in sentence:
                f.write(word + "\n")
            f.write("\n")
