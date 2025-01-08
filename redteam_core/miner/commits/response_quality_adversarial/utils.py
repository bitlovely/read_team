def find_matching_question(pattern):  
    file_path = 'questions.txt'  
    # Replace "BLANK" in the pattern with a regex wildcard  
    search_pattern = pattern.replace("BLANK", ".*")  

    # Use regex to match the pattern in the questions  
    import re  
    regex = re.compile(search_pattern, re.IGNORECASE)  

    try:  
        with open(file_path, 'r') as file:  
            for line in file:  
                question = line.strip()  
                if regex.fullmatch(question):  
                    return question  
    except FileNotFoundError:  
        print(f"The file {file_path} was not found.")  
        return None  

    return None  


# Pattern to search for  
pattern = "who wants to be a BLANK new zealand"  

# Finding the question  
matched_question = find_matching_question(pattern)  

if matched_question:  
    print(f"Matched Question: {matched_question}")  
else:  
    print("No matching question found.")  