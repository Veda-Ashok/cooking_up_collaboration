# Overcooked Agent Mod

## Building
Replace the game directory in all the hint paths in `overcooked_mod_mono.csproj` with your game path.

Install dotnet https://dotnet.microsoft.com/en-us/download

Run:
```bash
dotnet build -c Release
```

The DLL will be placed in the BepinEx/Plugin folder in the game directory.

## Writing Harmony Hooks
In `AgentMod.cs` we can add hooks to Unity `GameObject`s in Overcooked.

Finding the name of the class is the hardest part. Here we use dnspy to inspect the `Assembly-CSharp.dll` from the game directory.