"""
Detection Testing Script.

This scripts reads a given config file and runs the evaluation.
It is an entry point that is made to evaluate standard models in FsDet.

In order to let one script support evaluation of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import json
import os
import time

import detectron2.utils.comm as comm
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import launch
from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup
from fsdet.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    verify_results,
)

from groundingdino.util.inference import load_gdino_model
from load_models import load_fully_supervised_trained_model, load_clip_model
from utils import get_text_prompt_list_for_g_dino_with_classes, get_coco_to_lvis_mapping

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(
                COCOEvaluator(dataset_name, cfg, True, output_folder)
            )
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


class Tester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = Trainer.build_model(cfg)
        self.check_pointer = DetectionCheckpointer(
            self.model, save_dir=cfg.OUTPUT_DIR
        )

        self.best_res = None
        self.best_file = None
        self.all_res = {}

    def test(self, ckpt):
        self.check_pointer._load_model(self.check_pointer._load_file(ckpt))
        print("evaluating checkpoint {}".format(ckpt))
        res = Trainer.test(self.cfg, self.model)

        if comm.is_main_process():
            verify_results(self.cfg, res)
            print(res)
            if (self.best_res is None) or (
                self.best_res is not None
                and self.best_res["bbox"]["AP"] < res["bbox"]["AP"]
            ):
                self.best_res = res
                self.best_file = ckpt
            print("best results from checkpoint {}".format(self.best_file))
            print(self.best_res)
            self.all_res["best_file"] = self.best_file
            self.all_res["best_res"] = self.best_res
            self.all_res[ckpt] = res
            os.makedirs(
                os.path.join(self.cfg.OUTPUT_DIR, "inference"), exist_ok=True
            )
            with open(
                os.path.join(self.cfg.OUTPUT_DIR, "inference", "all_res.json"),
                "w",
            ) as fp:
                json.dump(self.all_res, fp)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    #VOC_CLASSES_WRONG_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']  # fmt: skip
    VOC_CLASSES = ['aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair', 'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor', 'bird', 'bus', 'cow', 'motorbike', 'sofa']
    #VOC_CLASSES = ['diningtable']
    COCO_CLASSES_ALL = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    #COCO_CLASSES_NOVEL = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair', 'couch', 'potted plant', 'dining table', 'tv']
    #COCO_CLASSES_BASE = ['truck','traffic light','fire hydrant','stop sign','parking meter','bench','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','bed','toilet','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
    LVIS_CLASSES = ['aerosol can','air conditioner','airplane','alarm clock','alcohol','alligator','almond','ambulance','amplifier','anklet','antenna','apple','applesauce','apricot','apron','aquarium','arctic (type of shoe)','armband','armchair','armoire','armor','artichoke','trash can','ashtray','asparagus','atomizer','avocado','award','awning','ax','baboon','baby buggy','basketball backboard','backpack','handbag','suitcase','bagel','bagpipe','baguet','bait','ball','ballet skirt','balloon','bamboo','banana','band aid','bandage','bandanna','banjo','banner','barbell','barge','barrel','barrette','barrow','baseball base','baseball','baseball bat','baseball cap','baseball glove','basket','basketball','bass horn','bat (animal)','bath mat','bath towel','bathrobe','bathtub','batter (food)','battery','beachball','bead','bean curd','beanbag','beanie','bear','bed','bedpan','bedspread','cow','beef (food)','beeper','beer bottle','beer can','beetle','bell','bell pepper','belt','belt buckle','bench','beret','bib','bible','bicycle','visor','billboard','binder','binoculars','bird','birdfeeder','birdbath','birdcage','birdhouse','birthday cake','birthday card','pirate flag','black sheep','blackberry','blackboard','blanket','blazer','blender','blimp','blinker','blouse','blueberry','gameboard','boat','bob','bobbin','bobby pin','boiled egg','bolo tie','deadbolt','bolt','bonnet','book','bookcase','booklet','bookmark','boom microphone','boot','bottle','bottle opener','bouquet','bow (weapon)','bow (decorative ribbons)','bow-tie','bowl','pipe bowl','bowler hat','bowling ball','box','boxing glove','suspenders','bracelet','brass plaque','brassiere','bread-bin','bread','breechcloth','bridal gown','briefcase','broccoli','broach','broom','brownie','brussels sprouts','bubble gum','bucket','horse buggy','horned cow','bulldog','bulldozer','bullet train','bulletin board','bulletproof vest','bullhorn','bun','bunk bed','buoy','burrito','bus (vehicle)','business card','butter','butterfly','button','cab (taxi)','cabana','cabin car','cabinet','locker','cake','calculator','calendar','calf','camcorder','camel','camera','camera lens','camper (vehicle)','can','can opener','candle','candle holder','candy bar','candy cane','walking cane','canister','canoe','cantaloup','canteen','cap (headwear)','bottle cap','cape','cappuccino','car (automobile)','railcar (part of a train)','elevator car','car battery','identity card','card','cardigan','cargo ship','carnation','horse carriage','carrot','tote bag','cart','carton','cash register','casserole','cassette','cast','cat','cauliflower','cayenne (spice)','cd player','celery','cellular telephone','chain mail','chair','chaise longue','chalice','chandelier','chap','checkbook','checkerboard','cherry','chessboard','chicken (animal)','chickpea','chili (vegetable)','chime','chinaware','crisp (potato chip)','poker chip','chocolate bar','chocolate cake','chocolate milk','chocolate mousse','choker','chopping board','chopstick','christmas tree','slide','cider','cigar box','cigarette','cigarette case','cistern','clarinet','clasp','cleansing agent','cleat (for securing rope)','clementine','clip','clipboard','clippers (for plants)','cloak','clock','clock tower','clothes hamper','clothespin','clutch bag','coaster','coat','coat hanger','coatrack','cock','cockroach','cocoa (beverage)','coconut','coffee maker','coffee table','coffeepot','coil','coin','colander','coleslaw','coloring material','combination lock','pacifier','comic book','compass','computer keyboard','condiment','cone','control','convertible (automobile)','sofa bed','cooker','cookie','cooking utensil','cooler (for food)','cork (bottle plug)','corkboard','corkscrew','edible corn','cornbread','cornet','cornice','cornmeal','corset','costume','cougar','coverall','cowbell','cowboy hat','crab (animal)','crabmeat','cracker','crape','crate','crayon','cream pitcher','crescent roll','crib','crock pot','crossbar','crouton','crow','crowbar','crown','crucifix','cruise ship','police cruiser','crumb','crutch','cub (animal)','cube','cucumber','cufflink','cup','trophy cup','cupboard','cupcake','hair curler','curling iron','curtain','cushion','cylinder','cymbal','dagger','dalmatian','dartboard','date (fruit)','deck chair','deer','dental floss','desk','detergent','diaper','diary','die','dinghy','dining table','tux','dish','dish antenna','dishrag','dishtowel','dishwasher','dishwasher detergent','dispenser','diving board','dixie cup','dog','dog collar','doll','dollar','dollhouse','dolphin','domestic ass','doorknob','doormat','doughnut','dove','dragonfly','drawer','underdrawers','dress','dress hat','dress suit','dresser','drill','drone','dropper','drum (musical instrument)','drumstick','duck','duckling','duct tape','duffel bag','dumbbell','dumpster','dustpan','eagle','earphone','earplug','earring','easel','eclair','eel','egg','egg roll','egg yolk','eggbeater','eggplant','electric chair','refrigerator','elephant','elk','envelope','eraser','escargot','eyepatch','falcon','fan','faucet','fedora','ferret','ferris wheel','ferry','fig (fruit)','fighter jet','figurine','file cabinet','file (tool)','fire alarm','fire engine','fire extinguisher','fire hose','fireplace','fireplug','first-aid kit','fish','fish (food)','fishbowl','fishing rod','flag','flagpole','flamingo','flannel','flap','flash','flashlight','fleece','flip-flop (sandal)','flipper (footwear)','flower arrangement','flute glass','foal','folding chair','food processor','football (american)','football helmet','footstool','fork','forklift','freight car','french toast','freshener','frisbee','frog','fruit juice','frying pan','fudge','funnel','futon','gag','garbage','garbage truck','garden hose','gargle','gargoyle','garlic','gasmask','gazelle','gelatin','gemstone','generator','giant panda','gift wrap','ginger','giraffe','cincture','glass (drink container)','globe','glove','goat','goggles','goldfish','golf club','golfcart','gondola (boat)','goose','gorilla','gourd','grape','grater','gravestone','gravy boat','green bean','green onion','griddle','grill','grits','grizzly','grocery bag','guitar','gull','gun','hairbrush','hairnet','hairpin','halter top','ham','hamburger','hammer','hammock','hamper','hamster','hair dryer','hand glass','hand towel','handcart','handcuff','handkerchief','handle','handsaw','hardback book','harmonium','hat','hatbox','veil','headband','headboard','headlight','headscarf','headset','headstall (for horses)','heart','heater','helicopter','helmet','heron','highchair','hinge','hippopotamus','hockey stick','hog','home plate (baseball)','honey','fume hood','hook','hookah','hornet','horse','hose','hot-air balloon','hotplate','hot sauce','hourglass','houseboat','hummingbird','hummus','polar bear','icecream','popsicle','ice maker','ice pack','ice skate','igniter','inhaler','ipod','iron (for clothing)','ironing board','jacket','jam','jar','jean','jeep','jelly bean','jersey','jet plane','jewel','jewelry','joystick','jumpsuit','kayak','keg','kennel','kettle','key','keycard','kilt','kimono','kitchen sink','kitchen table','kite','kitten','kiwi fruit','knee pad','knife','knitting needle','knob','knocker (on a door)','koala','lab coat','ladder','ladle','ladybug','lamb (animal)','lamb-chop','lamp','lamppost','lampshade','lantern','lanyard','laptop computer','lasagna','latch','lawn mower','leather','legging (clothing)','lego','legume','lemon','lemonade','lettuce','license plate','life buoy','life jacket','lightbulb','lightning rod','lime','limousine','lion','lip balm','liquor','lizard','log','lollipop','speaker (stero equipment)','loveseat','machine gun','magazine','magnet','mail slot','mailbox (at home)','mallard','mallet','mammoth','manatee','mandarin orange','manger','manhole','map','marker','martini','mascot','mashed potato','masher','mask','mast','mat (gym equipment)','matchbox','mattress','measuring cup','measuring stick','meatball','medicine','melon','microphone','microscope','microwave oven','milestone','milk','milk can','milkshake','minivan','mint candy','mirror','mitten','mixer (kitchen tool)','money','monitor (computer equipment) computer monitor','monkey','motor','motor scooter','motor vehicle','motorcycle','mound (baseball)','mouse (computer equipment)','mousepad','muffin','mug','mushroom','music stool','musical instrument','nailfile','napkin','neckerchief','necklace','necktie','needle','nest','newspaper','newsstand','nightshirt','nosebag (for animals)','noseband (for animals)','notebook','notepad','nut','nutcracker','oar','octopus (food)','octopus (animal)','oil lamp','olive oil','omelet','onion','orange (fruit)','orange juice','ostrich','ottoman','oven','overalls (clothing)','owl','packet','inkpad','pad','paddle','padlock','paintbrush','painting','pajamas','palette','pan (for cooking)','pan (metal container)','pancake','pantyhose','papaya','paper plate','paper towel','paperback book','paperweight','parachute','parakeet','parasail (sports)','parasol','parchment','parka','parking meter','parrot','passenger car (part of a train)','passenger ship','passport','pastry','patty (food)','pea (food)','peach','peanut butter','pear','peeler (tool for fruit and vegetables)','wooden leg','pegboard','pelican','pen','pencil','pencil box','pencil sharpener','pendulum','penguin','pennant','penny (coin)','pepper','pepper mill','perfume','persimmon','person','pet','pew (church bench)','phonebook','phonograph record','piano','pickle','pickup truck','pie','pigeon','piggy bank','pillow','pin (non jewelry)','pineapple','pinecone','ping-pong ball','pinwheel','tobacco pipe','pipe','pistol','pita (bread)','pitcher (vessel for liquid)','pitchfork','pizza','place mat','plate','platter','playpen','pliers','plow (farm equipment)','plume','pocket watch','pocketknife','poker (fire stirring tool)','pole','polo shirt','poncho','pony','pool table','pop (soda)','postbox (public)','postcard','poster','pot','flowerpot','potato','potholder','pottery','pouch','power shovel','prawn','pretzel','printer','projectile (weapon)','projector','propeller','prune','pudding','puffer (fish)','puffin','pug-dog','pumpkin','puncher','puppet','puppy','quesadilla','quiche','quilt','rabbit','race car','racket','radar','radiator','radio receiver','radish','raft','rag doll','raincoat','ram (animal)','raspberry','rat','razorblade','reamer (juicer)','rearview mirror','receipt','recliner','record player','reflector','remote control','rhinoceros','rib (food)','rifle','ring','river boat','road map','robe','rocking chair','rodent','roller skate','rollerblade','rolling pin','root beer','router (computer equipment)','rubber band','runner (carpet)','plastic bag','saddle (on an animal)','saddle blanket','saddlebag','safety pin','sail','salad','salad plate','salami','salmon (fish)','salmon (food)','salsa','saltshaker','sandal (type of shoe)','sandwich','satchel','saucepan','saucer','sausage','sawhorse','saxophone','scale (measuring instrument)','scarecrow','scarf','school bus','scissors','scoreboard','scraper','screwdriver','scrubbing brush','sculpture','seabird','seahorse','seaplane','seashell','sewing machine','shaker','shampoo','shark','sharpener','sharpie','shaver (electric)','shaving cream','shawl','shears','sheep','shepherd dog','sherbert','shield','shirt','shoe','shopping bag','shopping cart','short pants','shot glass','shoulder bag','shovel','shower head','shower cap','shower curtain','shredder (for paper)','signboard','silo','sink','skateboard','skewer','ski','ski boot','ski parka','ski pole','skirt','skullcap','sled','sleeping bag','sling (bandage)','slipper (footwear)','smoothie','snake','snowboard','snowman','snowmobile','soap','soccer ball','sock','sofa','softball','solar array','sombrero','soup','soup bowl','soupspoon','sour cream','soya milk','space shuttle','sparkler (fireworks)','spatula','spear','spectacles','spice rack','spider','crawfish','sponge','spoon','sportswear','spotlight','squid (food)','squirrel','stagecoach','stapler (stapling machine)','starfish','statue (sculpture)','steak (food)','steak knife','steering wheel','stepladder','step stool','stereo (sound system)','stew','stirrer','stirrup','stool','stop sign','brake light','stove','strainer','strap','straw (for drinking)','strawberry','street sign','streetlight','string cheese','stylus','subwoofer','sugar bowl','sugarcane (plant)','suit (clothing)','sunflower','sunglasses','sunhat','surfboard','sushi','mop','sweat pants','sweatband','sweater','sweatshirt','sweet potato','swimsuit','sword','syringe','tabasco sauce','table-tennis table','table','table lamp','tablecloth','tachometer','taco','tag','taillight','tambourine','army tank','tank (storage vessel)','tank top (clothing)','tape (sticky cloth or paper)','tape measure','tapestry','tarp','tartan','tassel','tea bag','teacup','teakettle','teapot','teddy bear','telephone','telephone booth','telephone pole','telephoto lens','television camera','television set','tennis ball','tennis racket','tequila','thermometer','thermos bottle','thermostat','thimble','thread','thumbtack','tiara','tiger','tights (clothing)','timer','tinfoil','tinsel','tissue paper','toast (food)','toaster','toaster oven','toilet','toilet tissue','tomato','tongs','toolbox','toothbrush','toothpaste','toothpick','cover','tortilla','tow truck','towel','towel rack','toy','tractor (farm equipment)','traffic light','dirt bike','trailer truck','train (railroad vehicle)','trampoline','tray','trench coat','triangle (musical instrument)','tricycle','tripod','trousers','truck','truffle (chocolate)','trunk','vat','turban','turkey (food)','turnip','turtle','turtleneck (clothing)','typewriter','umbrella','underwear','unicycle','urinal','urn','vacuum cleaner','vase','vending machine','vent','vest','videotape','vinegar','violin','vodka','volleyball','vulture','waffle','waffle iron','wagon','wagon wheel','walking stick','wall clock','wall socket','wallet','walrus','wardrobe','washbasin','automatic washer','watch','water bottle','water cooler','water faucet','water heater','water jug','water gun','water scooter','water ski','water tower','watering can','watermelon','weathervane','webcam','wedding cake','wedding ring','wet suit','wheel','wheelchair','whipped cream','whistle','wig','wind chime','windmill','window box (for plants)','windshield wiper','windsock','wine bottle','wine bucket','wineglass','blinder (for horses)','wok','wolf','wooden spoon','wreath','wrench','wristband','wristlet','yacht','yogurt','yoke (animal equipment)','zebra','zucchini']
    
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        is_gdino_model = args.is_gdino
        
        if is_gdino_model:
            model = load_gdino_model("cfg/GroundingDINO/GDINO.py", args.checkpoint, args.is_sl, args.is_PT)        
        if args.eval_iter != -1:
            # load checkpoint at specified iteration
            ckpt_file = os.path.join(
                cfg.OUTPUT_DIR, "model_{:07d}.pth".format(args.eval_iter - 1)
            )
            resume = False
        else:
            # load checkpoint at last iteration
            ckpt_file = cfg.MODEL.WEIGHTS
            resume = True
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            ckpt_file, resume=resume
        )
        
        if is_gdino_model:
            device = 'cuda'
            model = model.to(device)
            tokenizer = model.tokenizer
            class_len_per_prompt = 81
            if args.data_source=='voc':
                dataset_classes = VOC_CLASSES
            elif args.data_source=='coco':
                dataset_classes = COCO_CLASSES_ALL
            elif args.data_source=='lvis':
                dataset_classes = LVIS_CLASSES
            text_prompt_list, positive_map_list = get_text_prompt_list_for_g_dino_with_classes(dataset_classes, tokenizer, class_len_per_prompt)
            #text_prompt_list, positive_map_list = get_text_prompt_list_for_g_dino_with_classes(COCO_CLASSES_NOVEL, tokenizer, class_len_per_prompt)
            res = Trainer.test(cfg, model, args, text_prompt_list, positive_map_list, dataset_classes)
        else:
            res = Trainer.test(cfg, model, args)
            
        if comm.is_main_process():
            verify_results(cfg, res)
            # save evaluation results in json
            os.makedirs(
                os.path.join(cfg.OUTPUT_DIR, "inference"), exist_ok=True
            )
            with open(
                os.path.join(cfg.OUTPUT_DIR, "inference", "res_final.json"),
                "w",
            ) as fp:
                json.dump(res, fp)
        return res
    elif args.eval_all:
        tester = Tester(cfg)
        all_ckpts = sorted(tester.check_pointer.get_all_checkpoint_files())
        for i, ckpt in enumerate(all_ckpts):
            ckpt_iter = ckpt.split("model_")[-1].split(".pth")[0]
            if ckpt_iter.isnumeric() and int(ckpt_iter) + 1 < args.start_iter:
                # skip evaluation of checkpoints before start iteration
                continue
            if args.end_iter != -1:
                if (
                    not ckpt_iter.isnumeric()
                    or int(ckpt_iter) + 1 > args.end_iter
                ):
                    # skip evaluation of checkpoints after end iteration
                    break
            tester.test(ckpt)
        return tester.best_res
    elif args.eval_during_train:
        tester = Tester(cfg)
        saved_checkpoint = None
        while True:
            if tester.check_pointer.has_checkpoint():
                current_ckpt = tester.check_pointer.get_checkpoint_file()
                if (
                    saved_checkpoint is None
                    or current_ckpt != saved_checkpoint
                ):
                    saved_checkpoint = current_ckpt
                    tester.test(current_ckpt)
            time.sleep(10)
    else:
        if comm.is_main_process():
            print(
                "Please specify --eval-only, --eval-all, or --eval-during-train"
            )


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    if args.eval_during_train or args.eval_all:
        args.dist_url = "tcp://127.0.0.1:{:05d}".format(
            np.random.choice(np.arange(0, 65534))
        )
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
