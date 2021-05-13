from num2words import num2words


REPLACEMENTS = {
    # ORDINALS
    "1º": "primeiro",
    "2º": "segundo",
    "3º": "terceiro",
    "4º": "quarto",
    "5º": "quinto",
    "6º": "sexto",
    "7º": "sétimo",
    "8º": "oitavo",
    "9º": "nono",
    "10º": "décima",
    "1ª": "primeira",
    "2ª": "segunda",
    "3ª": "terceira",
    "4ª": "quarta",
    "5ª": "quinta",
    "6ª": "sexta",
    "7ª": "sétima",
    "8ª": "oitava",
    "9ª": "nona",
    "10ª": "décima",
    # Others
    ":": " e ",
    "+": "mais",
    "kg": " kilograma",
    "%": " porcento",
    "°": " graus",
    "-": " ",
    "/": " ",
    "  ": " ",
    "[": "",
    "]": "",
    "_": "",
    ",": "",
    ".": "",
    "*": "",
    "'": "",
    "r$": ""}


def replace_numbers(text):
    text = " ".join([num2words(txt, lang="pt-br")
                    if txt.isdigit() else txt for txt in text.split(" ")])
    return text.translate({ord(k): None for k in "0123456789"})


def clean_text(text):
    for k, v in REPLACEMENTS.items():
        text = text.replace(k, v)
    text = replace_numbers(text)
    text = text.strip()
    text = text.lower()
    return text


def clean_text_series(text_series):
    # Fix values
    for k, v in REPLACEMENTS.items():
        text_series = text_series.str.replace(k, v, regex=False)
    # Fix values that contains number
    text_series = text_series.apply(replace_numbers)
    text_series = text_series.str.strip().str.lower()
    return text_series
