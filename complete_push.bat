@echo off
cd /d c:\Users\User\curious_app
echo Aborting rebase...
git rebase --abort
echo.
echo Checking status...
git status
echo.
echo Pushing to GitHub with force...
git push -u origin main --force
echo.
echo Done! Your code is now on GitHub.
pause
