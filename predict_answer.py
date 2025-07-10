from llm_helper import llm

def generate_post(text):
    prompt = get_prompt(text)
    response = llm.invoke(prompt)
    return response.content


def get_prompt(text):
    prompt = f"""
        You are a helpful math tutor. Solve the following question step by step in plain English.

        DO NOT use LaTeX or math formatting. 
        Use only simple plain text. Write numbers like '3/4' or '1.5', not as fractions or powers.

        Here is the question:
        {text}

        Answer in this format:

        Step 1: ...
        Step 2: ...
        Step 3: ...
        Step 4: ...
        Step 5: ...
        """
    return prompt


if __name__ == "__main__":
    print(generate_post("Let $\\mathbf{a} = \\begin{pmatrix} -3 \\\\ 10 \\\\ 1 \\end{pmatrix},$ $\\mathbf{b} = \\begin{pmatrix} 5 \\\\ \\pi \\\\ 0 \\end{pmatrix},$ and $\\mathbf{c} = \\begin{pmatrix} -2 \\\\ -2 \\\\ 7 \\end{pmatrix}.$  Compute\\[(\\mathbf{a} - \\mathbf{b}) \\cdot [(\\mathbf{b} - \\mathbf{c}) \\times (\\mathbf{c} - \\mathbf{a})].\\]"))