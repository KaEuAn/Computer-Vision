# Meaning variables

LOWRES_SIZE = 500x500

# Usefull stuff

SOURCE_DIR = panorama_source
RESIZED_DIR = resized_$(LOWRES_SIZE)
LOWRES_DIR = lowres
HIGHRES_DIR = highres
NOT_RESIZED_DIR = not_resized

FILES_IN = $(wildcard $(SOURCE_DIR)/*)
FILES_OUT = $(FILES_IN:.png=.jpg)

LOWRES = $(subst $(SOURCE_DIR),$(LOWRES_DIR),$(FILES_OUT))
HIGHRES = $(subst $(SOURCE_DIR),$(HIGHRES_DIR),$(FILES_OUT))

# Pseudo-rules

all : lowres highres

lowres : $(LOWRES)
highres : $(HIGHRES)

clean :
	rm -rf "$(LOWRES_DIR)" "$(HIGHRES_DIR)" resized_* "$(NOT_RESIZED_DIR)"

.PHONY : all clean lowres highres

RESIZED_DIR LOWRES_DIR HIGHRES_DIR NOT_RESIZED_DIR :
	mkdir -p $($@)

# Lowres

$(RESIZED_DIR)/%.png : $(SOURCE_DIR)/%.jpg RESIZED_DIR
	gm convert -auto-orient -resize "$(LOWRES_SIZE)" "$<" "$@"

$(RESIZED_DIR)/%.png : $(SOURCE_DIR)/%.png RESIZED_DIR
	gm convert -auto-orient -resize "$(LOWRES_SIZE)" "$<" "$@"

$(LOWRES_DIR)/%.jpg : $(RESIZED_DIR)/%.png LOWRES_DIR
	stat "$@" >/dev/null || guetzli --quality 84 "$<" "$@"

# Highres

$(NOT_RESIZED_DIR)/%.png : $(SOURCE_DIR)/%.jpg NOT_RESIZED_DIR
	gm convert -auto-orient "$<" "$@"

$(NOT_RESIZED_DIR)/%.png : $(SOURCE_DIR)/%.png NOT_RESIZED_DIR
	cp "$<" "$@"

$(HIGHRES_DIR)/%.jpg : $(NOT_RESIZED_DIR)/%.png HIGHRES_DIR
	stat "$@" >/dev/null || guetzli --quality 84 "$<" "$@"
