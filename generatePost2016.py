import os
import shutil
from sgfmill import sgf, sgf_moves

target_dir = "pgd/Post2016"
afterAIfiles = []
game_files = os.popen("""find . -type f | grep '.sgf'""").read().split('\n')[:-1]
for filePath in game_files:
    if '.sgf' in filePath:
        parts = filePath.split("/")
        if len(parts) > 3:
            fileName = parts[3]
            year = fileName[:4]
            if year.isdigit():
                if int(year) >= 2016:
                    afterAIfiles.append(filePath)
                    shutil.move(filePath, target_dir + "/" + fileName)

