<div align="center">
    <h1>🌿 Watchplant 🌿</h1>
    <p>
      A fullstack app for graphing plant measurements.
    </p>
</div>
<div>
    <h3>
      Running the app:
    </h3>
</div>

1. Clone the git repository.
2. Navigate to the watchplant folder.
3. Activate a python virtual enviroment using `venv` or `conda`.
4. Install `pnpm`
5. Install python libraries using the comand:
```bash 
pip install -r server/requirements.txt
```
6. Install the libraries needed for the client program using the command:
```bash 
cd client && pnpm install
```
7. Run the following commands in separate terminals: 
```bash 
cd client && pnpm run dev
```
```bash 
cd server && uvicorn server.main:app --reload --host localhost
```
