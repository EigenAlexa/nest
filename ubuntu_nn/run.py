from nn_model import NNModel
def main():
    data = [(
         u"I'm using /set irc_conf_mode 1 in xchat but I still see notifications from people entering and exiting shouldn't those notifications be supressed ?",
         u'you have to restart xchat'), (u'you have to restart xchat', u'thanks!'), (
         u'OKAY my ubuntu crashed 5 times yesterday and twice already today.. just crashed again 1minute ago. what logs can i check to see waht the problem is!??! what logs can i check to see why my ubuntu crashed?!? does anyone know.. this is second time its crashed today, im really thinking about switching back to ubuntu i mean WINDOWS',
         u'there are logs in /var...'), (
         u"How do I run a command as another user? if the other user doesn't have a password?",
         u'sudo can do that tell us what you changed a pastebin of the sudoers file might help too..'),
         (u'!tell Gimp` about grub', u"you're too fast :\\"), (u"you're too fast :\\", u'no.  for edgy.'),
         (u'no.  for edgy.', u'right.'), (
         u'hello all. is there a way to get ubuntu-server 8.04 cd-image (not version 8.04.1)? there i can see only 8.04.1 images',
         u'Why do you want the old one? Why do you want 8.04?'), (
         u"ya, something about the protocol module in libgaim for some reason was called PRPL .. so they changed the name of libgaim to purple.. there was another library name change but i don't remember what it was. 1.5.0 is the latest 'stable' release of gaim .. but the 2.0.0 beta's are pretty stable themself.  cause the 1st release of pigdin just came out today as 2.0.0beta7",
         u'lol... gotcha'), (
         u'hola http://www.bandadels13.com/ descarga nuestra musica gratis my group music free download http://www.bandadels13.com/',
         u'This channel is not for advertising.'), (u'Ah! Mk It says 8.04 but im sure i downloaded 8.10 last nite :S',
                                                    u'right, you need to install libflashsupport for hardy :)'),
         (u'right, you need to install libflashsupport for hardy :)', u'u legend :D'), (
         u'is there any Ubuntu Linux driver for a Lexmark 8300 all-in-one printer? Can anyone help me?',
         u'You should be able to get PPD driver from Lexmark web site, I use CUPS do integrate them into our systems. http://support.lexmark.com:80/lexmark/index?page=content&productCode=&locale=EN&segment=SUPPORT&viewlocale=en_US&searchid=1344134341460&actp=search&userlocale=EN_US&id=DR860'),
         (u'auk: ok ty', u'Are you on 64bit?'), (u'Are you on 64bit?', u'no'),
         (u'no', u"OK, they're right... add repos"), (
         u"Hello all.  hopefully someone can help me I am running ubuntu 6.10.  I am very new to linux.... When I plug in a usb drive, it automounts, but it doesn't use very good names",
         u"what's your problem?"), (
         u'Define emergency for this n00b? Help? I have 3 NICs installed.  One onboard, a linksys, and an IBM.  From CLI, how do I determine which is the default in-use?',
         u'try them all? have u tried sudo apptitude ubuntu-desktop?'), (
         u"i'm having trouble getting mp3s to play in amarok.  it says i need to install mp3 support, and then, without it installing anything, it says support is installed and i need to restart amarok.  when i do, i just get the same error mp3s work in rythmbox however",
         u'go in synaptic and install every single gstreamer0.10 packages there is a #amarok , maybe they know'), (
         u'Hey, will there by any chance be a Windows Ubuntu Installer for 11.10 by any chance? When it does come out? I want to get a friend into Ubuntu.',
         u'wubi?')] # TODO replace with mongo iterator
    hyperparams = {
        'dimension' :  100,
        'window':  3,
        'min_count': 1,
        'workers': 4
    }
    # TODO check if model already exists and load it
    print("Setting up embedding")
    model = NNModel(sess=None,
                    conv_pairs=data,
                    store_sentences=True,
                    hyperparameters=hyperparams)
    print("Saving model")
    model.save()
    print("START TESTING")
    print("'Hi I'm Eigen. Let's talk'")
    while True:
        user_response = input()
        model_response = model.get_response(user_response)
        print(model_response)
if __name__ == "__main__":
    main()

