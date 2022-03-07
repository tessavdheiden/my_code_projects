# my_code_projects
This will be a repository of my code projects.

Create a tree, run one of these:

```
ipython create_tree.py
```
Output:
```
    __1__
   |     |
  2       3
 | |     | |
4   5   6   7
```
Output:
```
ipython create_general_tree.py
```

```
 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 
 |__|__|  |__|__|  |__|__|  |__|__|  |__|__|  |__|__|  |__|__|  |__|__|  |__|__|  
    5        6        7        8        9       10       11       12       13  
    |________|________|        |________|_______|        |________|________|   
             2                         3                          4           
             |_________________________|__________________________|           
                                       1                                     
```

Useful trick checking mistakes:
```
pylint create_general_tree.py
```

Useful trick to reformat:
```
autopep8 --in-place --aggressive --aggressive create_general_tree.py
```

Git commands for branching:
```
git checkout -b 'develop'
git add name_of_file_to_move_to_github second_file_to_move_to_github
git commit -m 'added new files'
git push --set -upstream origin develop
```

Once file is merged:
```
git checkout main
git pull
git branch -d develop
git branch
```