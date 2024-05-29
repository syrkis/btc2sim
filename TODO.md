# TODO

- [ ] Auto beautify tree. 2024-05-30
- [ ] Speedy playback animations (nice to have).
- [ ] Attach LMM to the website.
- [ ] Move from for loop to `jax.lax.scan`. 2024-05-30
- [ ] Select BT by hovering through MAP-Elites?
- [ ] Don't restart the episode when dead. 2024-05-30
- [ ] Test if the system fulfills the properties outlined in the paper.
- [ ] Make the environment more complex (move closer to Starcraft).
    - [ ] add maps with obstacles.
    - [ ] communication channel for units in sight from the same team
    - [ ] health packs or better healing units like in Starcraft (but would need to change Jaxmarl?)

# DONE

- [x] Compile tree from input. 2024-05-27
- [x] Run simulation on the website. 2024-05-27
- [x] Compute metrics live on the website. 2024-05-27
- [x] Ensure the streamlit app runs with Jaxmarl.
- [x] Parallel eval of trees.
