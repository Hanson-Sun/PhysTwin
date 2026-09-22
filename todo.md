- target move speed in models.json is not a good idea. We can just limit the controller move speed instead.

- try to look at the surface point and interior volume density


- genuinely remove texture and try again
- superglue pmo so much will need a different way to align the objects


- floor friction still a little wonky
- idk the rope thing really isnt learning that well... why? because of the interior points? 
    - lets clean up the parameters that we are learning...



- ok only thing missing now is the floor and object friction, is this learned anywhere?? maybe in the cma stage?



- rigid body not working? arent we supposed to have a dense spring mass system? this should be wayyy better at simulating rigid bodies whats going on??
    - missing bottom fgace since not shape prior
    - need volumetric dense representation somehow
    - maybe we add a new config for a rigid body?
    - we need a way to make this system more adaptable...
    - need better alignment system, look at digital twin v2. 
- take a look at the loss function, what is the criteria?

- do a simulation run where we grab a box with the claw and lift it up

- the rope didnt learn??
    - yeah the issue is with the contact resolution
    - i need to fix this, this is a fundamental learning issue

- please fixc the controller points visualization script to actually produce video, its kinda bad right now

- default should be visualization on!!! IMPORTANT



- generate more softbody types
- test pipeline
- integrate warp and mujoco sim
- modify warp simulation to be able to work with claw trajectory inputs
