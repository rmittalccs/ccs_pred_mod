# ccs_pred_mod

The basic assumption is that we'll be running this automation on the workbench. 

Here are the basic steps in order to get the project up and running on your workbench.

### Generate a public SSH (Secure Shell) key. 

[SSH keys are cryptographic keys used for authenticating and establishing secure connections between two systems over a network. You can then remote login or transfer files.]
1. Open a terminal in RStudio. Make sure you're in your home folder
  ```
  cd ~/
  ```  
2. Generate a ssh-key pair (replace the email with your own)
  ```
  ssh-keygen -t rsa -b 4096 -C "RMittal@ccsfundraising.com"
  ```
  This will have created a hidden folder in your home directory called .ssh which you can view by typing
  ```
  ls -a
  ```
3. Copy the contents of .ssh/id_rsa.pub into clipboard. You need to generate a new SSH key on github and save it.<br/>

  ***Very important:*** Copy the contents of the `.pub` file; the other one is your private key that only you should have it. **Do not share it with anybody else.**
  
  - On GitHub, go to settings.
  
      <img src="https://github.com/rmittalccs/ccs_pred_mod/assets/163910785/d1e76b03-2167-42f2-8fbb-cbadfb22bfe7" align="center" height="200">  
      <br/><br/>
      
  - Go to SSH and GPG keys.
  
      <img src="https://github.com/rmittalccs/ccs_pred_mod/assets/163910785/5d5ce358-9c3c-4f7a-b81d-5c8467664ac5" align="center" height="200">
      <br/><br/>
      
  - Click on "New SSH Key".
  
      <img src="https://github.com/rmittalccs/ccs_pred_mod/assets/163910785/1f5c57d5-aeff-4238-a13f-cc4afbc57fc7" align="center" height="200">
      <br/><br/>
  
  - Give it a meaningful title (e.g., CCS Posit Workbench) and paste the public SSH key into the box below.
  
      <img src="https://github.com/rmittalccs/ccs_pred_mod/assets/163910785/0c446a03-7fb5-426a-aa58-247e97d7f224" align="center" height="200">
      <br/><br/>
      

### Clone the git repository you need to git clone by typing the following in your terminal:

Open a terminal on RStudio or VSCode. Go to your home directory and type the following command. Enter "yes" when it ask you for a confirmation.

```
git clone git@github.com:rmittalccs/ccs_pred_mod.git
```
