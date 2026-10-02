we should test fixed clamp on object in warp (fixed to object points within radius, drop if outside radius, gate and release on squeeze) and see if that is good. 
check interior points? the controller points move right through the object

fix grip for EVERYTHING!! is there no contact friction?? what is going on
the test suite is kind of long lets fix that too


make lift trajectory hold on the top for a second before ending.
move front camera up a little more and point it down a little more.
investigate the grip strength, i see some strange teleportation effects. 
FIX TRELLIS SUPER GLUE MATCHING FOR SLOTH

WE NEED TO FIX THE GLB to SOFT BODY SIMULATION PIPELINE
TRAJECTORY IS STILL KIND OF WEIRD


- need to figure out how to handle damping?? where is it and how does warp handle it

- a bunch of things to fix i need to check if its clean and consistent
- the variable speed pathcing is still kind of ass need more robust collision boxes.
        - still some small issues, much better tho, ill just leave it for now until im back


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
