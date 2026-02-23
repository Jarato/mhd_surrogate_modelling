from autoenkoop import *
import os
import datetime

# making a new folder to save the script and the results 
script_name = os.path.basename(__file__)[:-3] # remove the ".py"
script_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = os.path.join(script_dir, "results", f"{script_name}", f"run_{timestamp}")
os.makedirs(folder_name, exist_ok=True)

# storing the script
with open(__file__, 'r') as file:
    script_content = file.read()
with open(os.path.join(folder_name, os.path.basename(__file__)), 'w') as file:
    file.write("\n"+"#"*100+"\n#\tTHIS SCRIPT HAS BEEN EXECUTED ALREADY\n#\tTHIS IS A COPY OF THE ORIGINAL SCRIPT\n#\tTHIS SCRIPT IS NOT MEANT TO BE EXECUTED AGAIN\n#\tIT EXISTS ONLY FOR THE PURPOSE OF GIVING CONTEXT TO THE DATA IN THIS FOLDER\n"+"#"*100+"\n\n"+script_content)


if __name__ == '__main__':
    print("hello world")