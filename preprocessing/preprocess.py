import pandas as pd
import numpy as np
import nltk
import swifter
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def pull_visits_containing_event(df, action):
    # create ind for action
    df['ind'] = df.action == action
    df['ind'] = df['ind'].astype(int)
    # tag sessions that contain indicator
    df['max_ind'] = df.groupby('visit_id')['ind'].transform(max)
    # subset data on sessions that contain indicator
    return_df = df[df.max_ind == 1]
    return(return_df)

def pull_close_events_to_action(df, action, n_closest=2):
    tmp = pd.DataFrame()
    df['keep'] = np.where(df.ind==1, df.sequence,np.nan)
    df['keep'] = df.groupby('visit_id')['keep'].transform(max)
    for v in df.visit_id.unique().tolist():
        v_df = df[(df.visit_id == v) & ((df.sequence >= df.keep - 2) & (df.sequence <= df.keep + 2))]
        tmp = tmp.append(v_df)
    return(tmp)


# replace nan values in url - needed to avoid exception in next function
def custom_function1(x):
    if 'globalsearch' in x.action:
        return('search')
    if ('"search"' in x.action) | (x.action =='click on "SÃ¸k"') | (x.action =='click on "Søk"'):
        return ('search')

    # key tasks
    if ('on "Activate SIM-Card"' in x.action) | ('on "Aktiver SIM-kort"' in x.action):
        return('activate_sim')
    if ('on "Aktiver"' in x.action) | ('on "Activate"' in x.action):
        return('click_on_activate')
    if ('on "Aktiver nytt"' in x.action) | ('on "Activate new"' in x.action):
        return('activate_new')
    if ('on "Send SMS til ansatte"' in x.action) | ('"Send SMS"' in x.action):
        return('send_sms_to_employees')
    if ('on "sms"' in x.action):
        return('click_on_sms')


   # 'click on "Search"' twice to complete task

    # new subs
    if ('on "Nytt abonnement"' in x.action) | ('on "Order new subscription"' in x.action) | ('on "Opprett nytt abonnement fra grunnen av"' in x.action):
        return ('new_sub')
    if 'on "Nytt abonnement eller eierskifte"' in x.action:
        return('click_on_new_subscription')
    if ('Bruk et eksisterende abonnement som mal' in x.action)| (x.action =='click on "Bruk malBruk et eksisterende abonnement som mal"'):
        return('click_new_subscription_from_template')
    if 'Nytt oppsettOpprett nytt abonnement fra grunnen av' in x.action:
        return('click_new_subscription_from_scratch')
    if x.action == 'click on "Abonnementstype"':
        return('click_on_subscription_type')
    if ('on "MobilAbonnement for tale og data"' in x.action) | ('on "Abonnement for tale og data"' in x.action):
        return ('click_on_sub_voice_data')
    if 'Ingen bindingstid' in x.action:
        return('select_no_binding')
    if 'Legg til binding' in x.action:
        return('select_binding')
    if x.action == 'on "Bruk eksisterende"' :
        return('click_on_use_existing')



    # change sub
    if (('subscriptions' in x.url) & ('_load_' in x.action)):
        return('loading subscriptions')
    if 'on "Endring av abonnement"' in x.action:
        return ('click_on_change_sub')
    if 'on "Administrer abonnement"' in x.action:
        return('click_on_manage_subscriptions')
    if ('on "subscription"' in x.action) | ('on "Subscription"' in x.action):
        return('click_on_subscription')
    if ('Eierskifte' in x.action):
        return('click_on_ownership')
    if 'on "Endre bruker av abonnement"' in x.action:
        return('click_on_change_user_on_sub')
    if 'on "Andre endringer"' in x.action:
        return('click_on_other_change')
    if 'on "Abonnement"' in x.action:
        return('click_on_subscription')
    if 'on "Mobil"' in x.action:
        return('click_on_mobil')
    if ('on "Abonnement"' in x.action) | ('on "abonnement"' in x.action):
        return('click_on_subscription')
    if ('on "Si opp"' in x.action) | ('on "Terminate"' in x.action):
        return('click_on_terminate')

    # order
    if ('Bestill' in x.action) | ('Order' in x.action) | (x.action == 'click on "Bestill nytt"'):
        return ('submit_order')
    if ('Bestill Si opp' in x.action): # -- interesting action
        return ('abort_order')
    if ('on "close"' in x.action):
        return ('interrupt_task')
    if 'on "Sjekk ut dette"' in x.action:
        return ('click_on_chek_out')

    # order history
    if 'on "Vis historikk"' in x.action:
        return('click_on_see_history')
    if x.action == 'on "Ferdig behandlet"':
        return('click_on_completed_processes')
    if x.action == 'on "Kansellert behandlet"':
        return('click_on_cancelled_processes')
    if 'orderhistory' in x.action:
        return('check_order_history')
    if ('on "Order overview"' in x.action) | (x.action=='click on "Ordreoversikt"'):
        return('click_on_order_overview')
    if (x.action == 'on "Page: UNDER ARBEID"'):
        return('check_status')

    # PUK code
    if ('on "Finn PUK-kode"' in x.action) | ('PUK' in x.action):
        return ('click_on_find_PUK')

    # Activate SIM
    if ('on "Activate SIM-Card"' in x.action) | ('on "Aktiver SIM-kort"' in x.action):
        return('activate_sim')

    # lock sim cards
    if ('on "lock"' in x.action) | ('on "Sperr"' in x.action) | ('SperretAktivt' in x.action):
        return('click_on_lock')
    if ('on "Locked"' in x.action) | ('on "Sperret"' in x.action):
        return('click_on_locked')

    # overview about cards
    if 'Datakort' in x.action:
        return('click_on_datacard')
    if ('on "SIM-kort"' in x.action) |('Hoved-SIM' in x.action):
        return('click_on_sim_card')
    if x.action =='click on "Aktivt"':
        return('click_on_activated')
    if (x.action == 'scroll on "Page: simcards"'):
        return('scroll_on_simcards')

    # account reference
    if 'on "Endre kontoreferanse"' in x.action:
        return ('click_on_change_account_reference')
    if 'Kontoreferanse' in x.action:
        return ('account_reference')
    if 'on "Ingen referanser"' in x.action:
        return ('select_on_no_reference')

    # Login
    if 'Magisk innlogging' in x.action:
        return ('click_on_magic_link')
    if 'on "Send engangskode"' in x.action:
        return ('click_on_send_entry_code')
    if 'on ""Verifiser kode"' in x.action :
        return('click_on_verify_code')

    # check subs
    if 'on "Mobilnummer"' in x.action:
        return ('click_on_mobilnummer')
    if ('on "Finn person"' in x.action) | ('on "Search for person"' in x.action):
        return ('click_on_find_person')
    if (x.action == 'scroll on "Page: subscriptions"'):
        return('scroll_on_subscriptions_page')

    # send SMS
    if 'on "Send"' in x.action:
        return ('click_on_send')

    # (additional) services
    if ('on "additional-services"' in x.action) | ('additionalServices' in x.action) | ('Tilleggstjenester' in x.action) :
        return ('adding_additional_services')
    if x.action == 'click on "Utforsk nye tjenester"':
        return ('explore_additional_services')
    if ('aktiv tjeneste' in x.action):
        return('click_on_active_services')
    if ('skytjeneste' in x.action) | ('Skytjeneste' in x.action):
        return('click_on_cloud')
    if 'on "APIs' in x.action:
        return ('click_on_APIs')
    if 'on "cloud"' in x.action :
        return('click_on_cloud')

    # agreements
    if ('on "Avtaler"' in x.action) | ('Avtaleprodukter' in x.action) | ('on "Agreements"' in x.action):
        return('click_on_agreements')
    if ('nytt avtaleprodukt' in x.action) | (x.action == 'click on "Legg til avtale"'):
        return('add_new_agreement')
    if 'on "Se avtale"' in x.action:
        return('see_agreements')

    # other
    if 'on "utland"' in x.action:
        return ('click_on_abroad')
    if 'on "navicon"' in x.action:
        return ('click_on_navicon')
    if 'on "pointer"' in x.action:
        return ('click_on_pointer')
    if 'on "Velg"' in x.action:
        return ('click_on_select')
    if ('on "Navn"' in x.action) | ('on "Name"' in x.action) :
        return ('click_on_name')
    if ('edit-thin' in x.action) | ('Editer' in x.action):
        return ('click_on_edit_field')
    if 'Konfigurer' in x.action:
        return ('click_on_configure')
    if 'Oppdater' in x.action:
        return ('click_on_update')
    if ('Lagre' in x.action) | ('Neste Lagre' in x.action) | ('Save' in x.action)| ('Next Save' in x.action):
        return ('click_on_save')
    if 'on "Legg til"' in x.action:
        return ('click_on_add')
    if ('trash' in x.action) | ('Slett' in x.action):
        return ('click_on_trash')
    if 'on "list"' in x.action:
        return ('click_on_list')
    if 'on "map"' in x.action:
        return ('click_on_map')
    if 'on "speach"' in x.action:
        return ('click_on_speach')
    if 'on "Sist bruk"' in x.action:
        return('click_on_last_used')
    if 'GDPR' in x.action:
        return('GDPR_related_click')
    if ('on "Aksepter og fortsett"' in x.action) | ('on "Accept and continue"' in x.action):
        return('click_on_accept_continue')
    if ('on "Fortsett"' in x.action) | ('on "Continue"' in x.action):
        return('click_on_continue')
    if 'on "Avbryt"' in x.action:
        return('click_on_cancel')
    if 'Reklamasjon' in x.action :
        return ('return_product')
    if 'on "Utsett"' in x.action:
        return('click_on_stocking')
    if 'on "Referanser"' in x.action:
        return('click_on_reference')
    if 'varslingspunkt' in x.action:
        return('warnings_product')
    if ('postalCode' in x.action):
        return('click_on_add_post_code')
    if ('on "Bytt konto"' in x.action) | ('on "Change account"' in x.action):
        return ('click_on_change_account')
    if 'on "expander"' in x.action:
        return('click_on_expand')
    if 'on "fusion"' in x.action:
        return('click_on_fusion')
    if 'on "crown"' in x.action:
        return('click_on_crown')
    if 'on "Se oppsagte"' in x.action:
        return('click_on_see_dismissed')
    if 'on "telenor"' in x.action:
        return('click_on_telenor')
    if x.action == 'on "edit-thin"' :
        return('open_edit_box')
    if x.action == 'on "*"' :
        return('click_on_star')
    if ('on "Bekreft"' in x.action) | ('on "Ok"' in x.action) |('on "Valider"' in x.action) | ('on "Confirm"' in x.action) | ('on "OK"' in x.action):
        return('click_on_confirm')
    if 'on "plus' in x.action:
        return ('click_on_plus')
    if 'on "Utstyrsendring' in x.action:
        return ('click_on_equipment_change')
    if 'Rabatt forbruk innland' in x.action:
        return('click_on_discount_consumption_inland')
    if 'on "close"' in x.action:
        return('click_on_close')
    if 'on ""Vis alle"' in x.action :
        return('click_on_show_all')
    if 'PATH' in x.action: # not sure what this is
        return('click_on_path')
    if ((x.url is not np.nan) & (x.url == 'https://www.telenor.no/bedrift/minbedrift/beta/#/') | (x.url == 'https://www.telenor.no/bedrift/minbedrift/beta/') | (x.url == 'https://www.telenor.no/bedrift/minbedrift/beta/mobile-app.html#/')) & ("_load_" in x.action):
        return ('load_homepage')
    # ToDo: Should we exclude other loading events since it's not human actions?
    if ((x.url is not np.nan) & (x.url != 'https://www.telenor.no/bedrift/minbedrift/beta/#/') | (x.url != 'https://www.telenor.no/bedrift/minbedrift/beta/') | (x.url != 'https://www.telenor.no/bedrift/minbedrift/beta/mobile-app.html#/') | ('subscriptions' not in x.url)) & ("_load_" in x.action):
        return ('load_other_page')
    if ('scroll on "Page: https://www.telenor.no/bedrift/minbedrift/beta/"' in x.action) | (x.action == 'scroll on "Page: https://www.telenor.no/bedrift/minbedrift/beta/#/"') | (x.action == 'scroll on "Page: mobile-app.html"'):
        return ('scroll_on_homepage')
    if ('on "Min Bedrift"' in x.action) | (x.action == 'keypress <RETURN> on "Page: https://www.telenor.no/bedrift/minbedrift/beta/"') | (x.action == 'click on "Page: https://www.telenor.no/bedrift/minbedrift/beta/"'):
        return ('go_back_to_homepage')
    if 'on "TEXTAREA"' in x.action: # -- to be specified better if possible
        return('interact_with_pop_up_window')
    if 'on "Lokasjoner"' in x.action:
        return ('click_on_locations')
    if ('on "Pause"' in x.action):
        return ('click_on_pause')
    if ('on "Choose"' in x.action):
        return ('click_on_choose')
    if ('on "Kopier"' in x.action)| ('on "Copy"' in x.action):
        return ('click_on_copy')
    if 'on "external"' in x.action:
        return ('click_on_external')
    if 'on "Ikke relevant"' in x.action:
        return ('click_on_not_relevant')
    if 'on "userType"' in x.action:
        return ('click_on_user_type')
    if 'on "refresh"' in x.action:
        return ('click_on_refresh')
    if 'on "Totalt"' in x.action:
        return ('click_on_overall')
    if ('on "Neste"' in x.action) | ('on "arrow"' in x.action )| ('on "Next"' in x.action ):
        return ('click_next')
    if ('on "Logg ut"' in x.action) | ('on "Log out"' in x.action):
        return ('click_log_out')

    #if (x.action.str.actions('click on "edit-thin"')) or (x.action.str.actions('click on "Save"')) or (x.action.str.actions('click on "Lagre"')):
    #    return('completed_editing_subscription')

    # bugs/ feil
    if ('ng-untouched ng-valid ng-dirty ng-valid-parse' in x.action) | ('ng-valid ng-dirty ng-valid-parse ng-touched' in x.action) | ('search-input ng-untouched' in x.action):
        return ('empty_subselect_ignore')

    # Settings
    if ('on "NOR"' in x.action) | ('on "ENG"' in x.action):
        return ('change_language')
    if ('on "Innstillinger"' in x.action) |('on "Settings"' in x.action) :
        return('click_on_new_settings')

    # reports
    if ('scroll' in x.action) & (('reports' in x.action)|('rapporter' in x.action)):
        return ('scroll_on_reports_page')
    if ('Eksporter' in x.action) | ('Export' in x.action)| ('icon icon-excel' in x.action):
        return ('export_to_excel')
    if 'scroll on "Page: downloads"' in x.action:
        return ('scroll_on_page_downloads')
    if ('on "Avanserte rapporter"' in x.action) | ('on "Advanced reports"' in x.action):
        return ('click_on_advanced_reports')
    if ('on "Bestilte rapporter"' in x.action):
        return ('click_on_ordered_reports')
    if ('on "Hent rapport"' in x.action) | ( 'on "Retrieve report"' in x.action):
        return ('click_on_get_report')
    if ('on "Ny rapport"' in x.action) | ('on "New report"' in x.action):
        return ('click_on_new_report')
    if 'on "Send til e-post"' in x.action:
        return('click_on_send_to_email')
    if ('January' in x.action) | ('january' in x.action) | ('february' in x.action) | ('February' in x.action) | ('mars' in x.action) | ('Mars' in x.action)  | ('april' in x.action) | ('April' in x.action)  | ('june' in x.action) | ('June' in x.action) | ('july' in x.action) | ('july' in x.action) | ('august' in x.action) | ('August' in x.action) |  ('september' in x.action) | ('September' in x.action) |  ('october' in x.action) | ('October' in x.action)|  ('november' in x.action) | ('November' in x.action) |  ('december' in x.action) | ('December' in x.action):
        return('click_on_time_range')

    # dealer
    if 'Forhandlere' in x.action:
        return ('click_on_dealers')

    # invoice
    if 'on "Endre fakturareferanse"' in x.action:
        return('click_on_change_invoice_ref')
    if ('Fakturert' in x.action) | ('Invoiced' in x.action):
        return('click_on_billed')
    if ('on "ubetalte fakturaer"' in x.action) | ('unpaidinvoices' in x.action):
        return ('click_on_unpaid_invoices')
    if ('Faktureringsinformasjon' in x.action):
        return ('click_on_billing_information')
    if 'on "Betalt"' in x.action:
        return ('click_on_paid')
    if 'on "Se siste fakturaer"' in x.action:
        return ('see_last_invoices')
    if ('on "Fakturakonto -"' in x.action) | ('Fakturakonto' in x.action) | ('Fakturanummer' in x.action) | ('fakturakontoer"' in x.action):
        return ('open_invoice_account')
    if 'on "Ny Fakturakonto"' in x.action:
        return ('click_on_new_invoice_account')
    if ('on "Bestill fakturakopi"' in x.action) | ('on "Bestill kopi"' in x.action) | ('on "Order copy"' in x.action):
        return ('order_invoice_copy')
    if 'scroll on "Page: costs"' in x.action:
        return('scroll_on_page_costs')

    # administrators
    if ('Administratorer' in x.action) | ('Administrators' in x.action):
        return('click_on_administrators')
    if ('inaktive administrator' in x.action) | ('inactive administrator' in x.action):
        return('click_on_inactive_administrators')
    if 'on "Rolle og avdeling"' in x.action:
        return('click_on_role_department')
    if 'on "Gi samme tilgang som meg"' in x.action:
        return('click_on_give_the_same_access')
    if ('on "Legg til administrator"' in x.action) | ('on "Ny administrator"' in x.action):
        return('click_on_add_admin')
    if ('on "Slett administrator"' in x.action) | ('on "Delete administrator"' in x.action):
        return('click_on_delete_admin')
    if 'on "Gi tilgang"' in x.action:
        return('click_on_give_access')

    # administrate companies
    if ('Selskap' in x.action) | ('on "Companies"' in x.action) | ('on "company"' in x.action) | ('Companies' in x.action) | ('Selskaper' in x.action):
        return('click_on_company')
    if 'on "egg til Selskaper"' in x.action:
        return('click_on_add_companies')
    if ('on "account"' in x.action) | ('Account' in x.action) | ('Kontoer' in x.action):
        return ('click_on_account')
    if x.action =='scroll on "Page: accounts"':
        return ('scroll_on_accounts_page')

    # product specific
    if x.action == 'on "Fastnett"':
        return('click_on_fixed')
    if x.action == 'on "Ingen datapakke inkludert"':
        return('click_on_no_data_incl')
    if ('on "data"' in x.action) | ('on "Datapakke"' in x.action) | ('datapakker' in x.action):
        return('click_on_data')
    if ('on "Mobilt BedriftsnettKonfigurerSe avtale"' in x.action):
        return('click_on_MBN_agreements')
    if (x.action == 'on "Mobilt Bedriftsnett"') | (x.action == 'on "mbn"'):
        return('click_on_MBN')
    if 'Bedrift' in x.action:
        return ('click_on_bedrift_product')

    # help
    if 'on "chat"' in x.action:
        return('start_chat')
    if ('on "Kontakt oss"' in x.action) | ('on "support"' in x.action) | (x.action =='click on "Vi hjelper deg!"') | ('hjelp' in x.action):
        return('click_on_contact_us')
    if 'Feil' in x.action:
        return('click_on_error_details')
    if 'on "question"' in x.action:
        return('click_on_question')
    if 'information' in x.action:
        return('click_on_information')

    # notifications
    if 'Varslinger' in x.action:
        return('click_on_notifications')
    if 'Meldinger fra Telenor' in x.action:
        return('click_on_inbox')
    if 'anbefaling' in x.action:
        return('click_on_recommendations')
    else:
        return('click_on_other')

def process1(i):
    if isinstance(i,float): # in case of nans (No longer needed if you deleted them already)
        pass
    else:
        i = nltk.word_tokenize(i)
        i = nltk.pos_tag(i)
        return i

def process2(row, output_format=None):
    d = process1(row)
    d1 = [('number', x[1]) if x[1] == 'CD' else x for x in d]
    if output_format == 'entities':  # return action split into components
        return(d1)
    else:  # returns replaced action with number replaced with "number"
        d2 = " ".join([ent[0] for ent in d1]).replace('``', '"').replace("''", '"')
        return(d2)

def custom_function3(x):
    #### action1 tagging
    if (x.action_cleaned == 'click_on_other') & (('on " number number number "' in x.action_numbers_repl) | ('on " number "' in x.action_numbers_repl) | ('on " Abonnement ( number ) "' in x.action_numbers_repl)):
        return ('click_on_number')
    if ('scroll on " Page: number "' in x.action_numbers_repl) | ('scroll on " number Abonnement "' in x.action_numbers_repl):
        return ('scroll_on_number')
    else:
        return(x.action_cleaned)

def clean_actions(t):

    action = 'click on "PATH"'
    action = 'click on "Search"'

    t['ind'] = t.action == action
    t['ind'] = t['ind'].astype(int)

    test = pull_visits_containing_event(t,action)
    # look at examples
    test[['user','sequence','url','action']].sort_values(['user','sequence'])

    # apply manual tagging function - REALLY SLOW :P
    # t['action_cleaned'] = t.apply(custom_function1, axis=1)
    # check which events are untagged and most frequent (candidates for manual tagging)
    # t[t['action_cleaned']=='click_on_other'].action.value_counts()
    ########### run in parallel
    t['action_cleaned'] = t.swifter.apply(custom_function1, axis=1)
    # inspect unlassified actions
    t[t['action_cleaned']=='click_on_other'].action.value_counts()
    ###########################################################################
    # identify and replace number entities


    t['action_numbers_repl'] = t.action.swifter.apply(lambda x: process2(x))
    ###########################################################################
    # manual tagging 2: extends "action_cleaned" to include actions that contained "numbers"
    t[(t['action_cleaned']=='click_on_other')].action_numbers_repl.value_counts()



    t['action_cleaned'] =  t.swifter.apply(custom_function3, axis=1)
    ###########################################################################
    # manual tagging 3: extends "action_cleaned" to include other actions with high frequency
    t[(t['action_cleaned']=='click_on_other')].action.value_counts()

    return t
