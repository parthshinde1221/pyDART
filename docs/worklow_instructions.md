# Summary of the Development Workflow Process

## Development workflow steps



1. **Pull the latest changes from the `main` branch:**
   ```bash
    git checkout main
    git pull origin main

2. **Pull the latest changes from the `research` branch:**
   ```bash
    git checkout research
    git pull origin research

3. **Decide which branch to work on:**
    - Use `research` for experimental work
    - Use `main` for stable, production-ready changes

**4. Optionally, create a new feature branch to isolate your work**
    Uncomment and replace 'your-feature-name' to create a new feature branch
    ```bash
    git checkout -b feature/your-feature-name

**5. Make your changes and thoroughly test them**
    (Edit, add, or remove files as needed)

**6. Add your changes to the staging area**
    ```bash
    git add .

**7. Commit your changes with a meaningful message**
    ```bash
    git commit -m "Describe the feature or changes made"

**8. Push your changes to the remote repository**
    Replace 'feature/your-feature-name' with your branch name if you used one
    ```bash
    git push origin feature/your-feature-name

**9. (Optional) Create a Pull Request on GitHub for code review**

**10. Merge your feature branch back into `research` or `main`:**
    ```bash
    # Replace 'feature/your-feature-name' with your branch name if you used one
    ```bash
    git checkout research  # or 'main'
    git merge feature/your-feature-name

**11. Clean up the feature branch after merging (if you used one)**
    Replace 'feature/your-feature-name' with your branch name
    ```bash
    git branch -d feature/your-feature-name

**12. Push the merged branch to GitHub**
    Replace 'research' with 'main' if you merged into main
    ```bash
    git push origin research

