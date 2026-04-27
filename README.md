# Reel-ize Movie Finder

Reel-ize movie finder is like the "Shazam", but for movies. By uploading any still picture from a scene of a movie with at least one actor in it, our application will call our custom age and facial-detection models to try to predict the actors, actor ages, and movie that the scene is from. 

The age model is built using a ResNet50V2 regressor backbone with a custom CNN head. The facial recognition model is built upon ArcFace and DeepFace.

With our preliminary testing, we found that the application is accurate on some stills (especially ones with more than one actor is can positively identify), but inaccurate on others. These are due to underfitting of our age and facial models.

Website: [https://gitops-mk.opensource.mieweb.org/](https://gitops-mk.opensource.mieweb.org/)


## Steps for Using Application

1. Here is the page that the link above will bring you to. You have the option to upload an image or video, along with having the option to drag and drop an image or video.
<img width="935" height="843" alt="Screenshot 2026-04-27 at 6 49 07 PM" src="https://github.com/user-attachments/assets/c3c76760-101f-44ed-83cf-be1d011338ff" />


2. Once the image or video is uplaoded, the user will press the run analysis button.


3. Here you will see that the image or video is processing.
   <img width="925" height="694" alt="Screenshot 2026-04-27 at 6 33 53 PM" src="https://github.com/user-attachments/assets/3363045a-d1fc-4c50-80fe-1b3f2f04e231" />


4. Once the image or video is finished processing, you should see the results of the face/s detetced and actor/s found with possible movies that the actor/s appear in.


<img width="935" height="843" alt="Screenshot 2026-04-27 at 6 42 29 PM" src="https://github.com/user-attachments/assets/a5307233-2262-4e3e-bcde-13fae6f03c6a" />

<img width="935" height="843" alt="Screenshot 2026-04-27 at 6 43 06 PM" src="https://github.com/user-attachments/assets/4bb7b2bb-be61-45f7-adca-2ec14131fd30" />


5. If you click on the predicted actor that the website returns, it will reveal the age prediction of an actor along with the top 3 predicted actors that the API thought that this person could be.
  <img width="935" height="843" alt="Screenshot 2026-04-27 at 6 44 21 PM" src="https://github.com/user-attachments/assets/0552ce20-8721-4ff3-b363-cc1edc226ed5" />