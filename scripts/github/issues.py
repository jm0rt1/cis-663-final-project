from github import Github
import re
from pathlib import Path
import sys

# Adding the grandparent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))  # nopep8

from scripts.github.access_token import ACCESS_TOKEN  # nopep8


# Initialize using an access token
g = Github(ACCESS_TOKEN)

# Name of your repository
REPO_NAME = 'jm0rt1/cis-663-final-project'

# Fetch the repo
repo = g.get_repo(REPO_NAME)

# Parse the markdown file


def parse_md(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    milestones = re.split(r'##\s+', content)[1:]
    parsed_data = {}

    for m in milestones:
        lines = m.splitlines()
        milestone_name = lines[0].strip()

        issue_data = []
        for i in lines[1:]:
            if not i:
                continue
            issue_title = i.split(']')[1].split('|')[0].strip()
            assignee = None
            labels = None

            if len(i.split('|')) > 1:
                assignee = i.split('|')[1].strip()

            if len(i.split('|')) > 2:
                labels = [label.strip()
                          for label in i.split('|')[2].split(',')]

            issue_data.append((issue_title, assignee, labels))

        parsed_data[milestone_name] = issue_data

    return parsed_data

# Create milestones and issues


def create_milestones_and_issues(data):
    for milestone_name, issues in data.items():
        # Create or fetch milestone
        milestone = None
        for m in repo.get_milestones():
            if m.title == milestone_name:
                milestone = m
                break

        if milestone is None:
            milestone = repo.create_milestone(milestone_name)

        # Create issues
        for issue_tuple in issues:
            issue_title, assignee, labels = issue_tuple
            repo.create_issue(title=issue_title, milestone=milestone,
                              assignee=assignee, labels=labels)


data = parse_md('docs/tasks/tasks.md')
create_milestones_and_issues(data)
