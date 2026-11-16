# Ported from ModelScope/EvalScope, Apache-2.0; see ../NOTICE.md.
# Copyright LiveCodeBench @ 2024,


def extract_code_generation(model_output: str, model_type: str = "chat"):
    # modified from
    outputlines = model_output.split("\n")
    # TODO: handle codellama

    if model_type == "base":
        return model_output.strip()
    elif model_type == "chat":
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    else:
        raise ValueError(f"Invalid mode type: {model_type}")

    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])
