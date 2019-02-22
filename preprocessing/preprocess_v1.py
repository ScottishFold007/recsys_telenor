import pandas as pd
import numpy as np
import nltk
import swifter

in_path = '//tns-fbu-2f-117/user-117$/t914868/Mine dokumenter/dynatrace/files_for _msc_project/'
t = pd.read_csv(in_path+'splunk_data_180918_aksel.txt',encoding='UTF-8', dtype={"user_id": int, "visit_id": int, "sequence": int, "start_time":object, "event_duration":float,"url":str, "action":str, "country":str,"user_client":str,"user_client_family":str,"user_experience":str,"user_os":str,"apdex_user_experience":str,"bounce_rate":float,"session_duration":float})
t.columns = t.columns.str.replace('min_bedrift_event.','')
t = t[~t.action.isnull()]
# replace nan values in url - needed to avoid exception in next function
t.url = t.url.fillna('placeholder')

####################################################
def tagging1(x):

    # search
    if 'globalsearch' in x.action:
        return('search')
    if ('"search"' in x.action) | ('"Search"' in x.action) | ('"Søk"' in  x.action) | (x.action=='click on "SÃ¸k"') | ('keypress <RETURN> on "Page: https://www.telenor.no/bedrift/minbedrift/beta/"' in x.action) | (x.action =='keypress <RETURN> on "Page: https://www.telenor.no/bedrift/minbedrift/beta/#/"') | (x.action == 'keypress <RETURN> on "searchKey"') | (x.action == 'keypress <RETURN> on "SÃ¸k"'):
        return ('search')

    # activate SIM-card
    if ('on "Activate SIM-Card"' in x.action) | ('on "Aktiver SIM-kort"' in x.action):
        return('activate_sim')
    if ('on "Aktiver"' in x.action) | ('on "Activate"' in x.action):
        return('click_on_activate')
    if ('on "Aktivert"' in x.action) | ('on "Activated"' in x.action):
        return('click_on_activated')
    if ('on "Aktiver nytt"' in x.action) | ('on "Activate new"' in x.action):
        return('activate_new')
    if ('on "Send SMS til ansatte"' in x.action) | ('"Send SMS"' in x.action):
        return('send_sms_to_employees')
    if ('on "sms"' in x.action):
        return('click_on_sms')

   # 'click on "Search"' twice to complete task

    # new subs
    if (x.action == 'click on "New subscription"') | ('on "Order new subscription"' in x.action) | ('on "Opprett nytt abonnement fra grunnen av"' in x.action) | ("New setupCreate new subscription" in x.action) | ('on "Nytt abonnement eller eierskifte"' in x.action) | ('on "Nytt abonnement"' in x.action) | ('"Create new subscription from scratch"' in x.action) :
        return ('click_on_new_subscription')
    if ('on "Neste"' in x.action) | ('on "Next"' in x.action ) | ('on "Fortsett"' in x.action ) | ('on "Continue"' in x.action ):
        return ('click_next_in_subscr_order')
    if ('on "Choose"' in x.action):
        return ('click_on_choose_in_subscr_order')
    if ('Bruk et eksisterende abonnement som mal' in x.action)| ('Bruk malBruk et eksisterende abonnement som mal' in x.action) | ('click on "Use templateUse an existing subscrip' in x.action) | ('"Bruk mal"' in x.action):
        return('click_new_subscription_from_template')
    if ('Nytt oppsettOpprett nytt abonnement fra grunnen av' in x.action) | (x.action == 'click on "Nytt oppsett"'):
        return('click_new_subscription_from_scratch')
    if 'Abonnementstype' in x.action:
        return('click_on_subscription_type')
    if ('on "MobilAbonnement for tale og data"' in x.action) | ('on "Abonnement for tale og data"' in x.action) | (x.action == 'click on "MobileSubscription for voice and data"'):
        return ('click_on_sub_voice_data')
    if ('Ingen bindingstid' in x.action) | ( x.action == 'click on "No lock-in"'):
        return('select_no_binding')
    if 'Legg til binding' in x.action:
        return('select_binding')
    if 'Bruk eksisterende' in x.action :
        return('click_on_use_existing')
    if 'ordercase' in x.action :
        return('click_on_ordercase')

    # change sub
    if (('subscriptions' in x.url) & ('_load_' in x.action)):
        return('loading_subscriptions')
    if (x.action == 'click on "Flere - Endring av abonnement"') :
        return ('click_on_change_sub')
    if 'on "User references"' in x.action:
        return('click_on_user_references')
    if 'on "Administrer abonnement"' in x.action:
        return('click_on_manage_subscriptions')
    if ('subscriptions' in x.action) | (x.action =='click on "Subscriptions"') | (x.action =='click on "abonnementer"')| (x.action =='click on "Subscription"') | (x.action =='click on "subscription"') | (x.action =='click on "Subscription Type"'):
        return('click_on_subscription')
    if ('Eierskifte' in x.action):
        return('click_on_ownership')
    if ('on "Endre bruker av abonnement"' in x.action) | (x.action == 'click on "Change user of subscription"'):
        return('click_on_change_user_on_sub')
    if 'on "Andre endringer"' in x.action:
        return('click_on_other_change')
    if 'on "Mobil"' in x.action:
        return('click_on_mobil')
    if ('on "Abonnement"' in x.action) | ('on "abonnement"' in x.action):
        return('click_on_subscription')
    if ('on "Si opp"' in x.action) | ('on "Terminate"' in x.action) | ('on "Cancel"' in x.action) | ('Oppsigelse' in x.action):
        return('click_on_terminate')

    # order
    if ('Bestill' in x.action) | ('Order' in x.action) | (x.action == 'click on "Bestill nytt"') | (x.action == 'click on "Ny bestilling"'):
        return ('submit_order')
    if ('Bestill Si opp' in x.action): # -- interesting action
        return ('abort_order')
    if ('on "close"' in x.action):
        return ('interrupt_task')
    if 'on "Sjekk ut dette"' in x.action:
        return ('click_on_check_out')

    # order overview
    if ('on "Ferdig behandlet"' in x.action) | ('on "Completed"' in x.action):
        return('order_overview_check_completed_orders')
    if ('on "In process"' in x.action) | ('on "Til behandling"' in x.action) | ('on "UNDER ARBEID"' in x.action):
        return('order_overview_check_orders_in_process')
    if ('on "Others"' in x.action) | ('on "Andre"' in x.action):
        return('order_overview_check_other_orders')
    if ('on "Single order"' in x.action) | ('on "Enkeltordre"' in x.action):
        return('order_overview_check_single_orders')
    if ('on "Bulk order"' in x.action) | ('on "Samleordre"' in x.action):
        return('order_overview_check_bulk_orders')
    if ('on "Change on subscription"' in x.action) | ('on "Endring av abonnement"' in x.action):
        return('order_overview_check_change_sub')
    if ('on "Transferred to new provider"' in x.action) | ('on "Overflytting til annen operator"' in x.action):
        return('order_overview_click_port_out')
    if ('on "Change account"' in x.action) | ('on "Endring av konto"' in x.action):
        return('order_overview_check_change_account')
    if ('on "Transferred to Telenor"' in x.action) | ('on "Overflyttet til Telenor"' in x.action):
        return('order_overview_check_port_ins')
    if ('on "Subscription blocked"' in x.action) | ('on "Abonnement sperret"' in x.action):
        return('order_overview_check_locked_subs')
    if ('Kansellert behandlet' in x.action) | (x.action == 'click on "Kansellert behandling"'):
        return('order_overview_check_cancelled_subs')
    if ('on "Send til e-post"' in x.action) | ('on "Send to e-mail"' in x.action):
        return('click_on_send_order_status_to_email')

    # order history
    if 'on "Vis historikk"' in x.action:
        return('click_on_see_history')
    if ('orderhistory' in x.action) | ('on "Order overview"' in x.action) | (x.action=='click on "Ordreoversikt"') | (x.action=='click on "See orders"') | (x.action == 'scroll on "Page: order"'):
        return('click_on_order_overview')
    if ('Page: UNDER ARBEID' in x.action):
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
    if ('on "Locked"' in x.action) | ('on "Sperret"' in x.action) | ('on "sim-barred"' in x.action):
        return('click_on_locked')
    if (x.action =='pne sperring') | ('" Ã pne sperring "' in x.action):
        return ('click_on_unlock')

    # overview about cards
    if 'Datakort' in x.action:
        return('click_on_datacard')
    if ('on "SIM-kort"' in x.action) |('Hoved-SIM' in x.action) | (x.action == 'click on "sim"') | (x.action == 'click on "SIM-cards"'):
        return('click_on_sim_card')
    if ('on "Tvilling-SIM"' in x.action) |('Twin-SIM' in x.action):
        return('click_on_twin_sim')
    if x.action =='click on "Aktivt"':
        return('click_on_activated')
    if (x.action == 'scroll on "Page: simcards"') | ('scroll on "Page: simcards' in x.action):
        return('scroll_on_simcards')

    # account reference
    if ('on "Endre kontoreferanse"' in x.action) | ('on "Endre referanser"' in x.action):
        return ('click_on_change_account_reference')
    if x.action == 'click on "Legg til konto"':
        return ('click_add_account')
    if 'Kontoreferanse' in x.action:
        return ('account_reference')
    if 'on "Ingen referanser"' in x.action:
        return ('select_on_no_reference')

    # Login
    if 'Magisk innlogging' in x.action:
        return ('click_on_magic_link')
    if 'on "Send engangskode"' in x.action:
        return ('click_on_send_entry_code')
    if 'on "Verifiser kode"' in x.action :
        return('click_on_verify_code')

    # check subs
    if ('on "Mobilnummer"' in x.action) | (x.action == 'click on "Mobilnummer *"'):
        return ('click_on_mobilnummer')
    if ('on "Finn person"' in x.action) | ('on "Search for person"' in x.action):
        return ('click_on_find_person')
    if (x.action == 'scroll on "Page: subscriptions"'):
        return('scroll_on_subscriptions_page')

    # send SMS
    if 'on "Send"' in x.action:
        return ('click_on_send')

    # agreements
    if ('on "Avtaler"' in x.action) | ('Avtaleprodukter' in x.action) | ('avtaleprodukter' in x.action) | ('on "Agreements"' in x.action) | ('agreementtypes' in x.action):
        return('click_on_agreements')
    if ('nytt avtaleprodukt' in x.action) | (x.action == 'click on "Legg til avtale"') |(x.action == 'click on "Opprette Avtalen"'):
        return('add_new_agreement')
    if ('on "Se avtale"' in x.action) | ('on "View agreement"' in x.action):
        return('see_agreements')

    # other
    if 'on "Ingen retningslinjer satt"' in x.action:
        return ('click_on_guidelines')
    if ('click on "Installed in"' in x.action) | ('click on "Installert i"' in x.action):
        return ('click_on_installed_in')
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
    if ('Oppdater' in x.action) | (x.action =='click on "Send oppdatering"'):
        return ('click_on_update')
    if ('Lagre' in x.action) | ('Neste Lagre' in x.action) | ('Save' in x.action)| ('Next Save' in x.action):
        return ('click_on_save')
    if ('on "Legg til"' in x.action) | (x.action== 'click on "Add"'):
        return ('click_on_add')
    if ('trash' in x.action) | ('Slett' in x.action):
        return ('click_on_trash')
    if 'on "list"' in x.action:
        return ('click_on_list')
    if 'on "map"' in x.action:
        return ('click_on_map')
    if 'on "Filtrer"' in x.action:
        return ('click_on_filter')
    if 'on "speach"' in x.action:
        return ('click_on_speach')
    if x.action =='click on "Sist brukt"':
        return('click_on_last_used')
    if 'GDPR' in x.action:
        return('GDPR_related_click')
    if ('on "Aksepter og fortsett"' in x.action) | ('on "Accept and continue"' in x.action):
        return('click_on_accept_continue')
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
    if ('on "Bytt konto"' in x.action) :
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
    if 'on "edit-thin"' in x.action :
        return('open_edit_box')
    if 'on "*"' in x.action :
        return('click_on_star')
    if ('on "Bekreft"' in x.action) | ('on "Ok"' in x.action) |('on "Valider"' in x.action) | ('on "Confirm"' in x.action) | ('on "OK"' in x.action) | ('on "Validate"' in x.action):
        return('click_on_confirm')
    if 'on "plus' in x.action:
        return ('click_on_plus')
    if 'on "Utstyrsendring' in x.action:
        return ('click_on_equipment_change')
    if 'Rabatt forbruk innland' in x.action:
        return('click_on_discount_consumption_inland')
    if 'on "close"' in x.action:
        return('click_on_close')
    if 'on "Vis alle"' in x.action :
        return('click_on_show_all')
    if 'PATH' in x.action: # not sure what this is
        return('click_on_path')
    if ((x.url is not np.nan) & (x.url == 'https://www.telenor.no/bedrift/minbedrift/beta/#/') | (x.url == 'https://www.telenor.no/bedrift/minbedrift/beta/') | (x.url == 'https://www.telenor.no/bedrift/minbedrift/beta/mobile-app.html#/')) & ("_load_" in x.action):
        return ('load_homepage')
    # ToDo: Should exclude other loading events since it's not human actions
    if ((x.url is not np.nan) & (x.url != 'https://www.telenor.no/bedrift/minbedrift/beta/#/') | (x.url != 'https://www.telenor.no/bedrift/minbedrift/beta/') | (x.url != 'https://www.telenor.no/bedrift/minbedrift/beta/mobile-app.html#/') | ('subscriptions' not in x.url)) & ("_load_" in x.action):
        return ('load_other_page')
    if ( x.action =='scroll on "Page: https://www.telenor.no/bedrift/minbedrift/beta/"') | (x.action == 'scroll on "Page: https://www.telenor.no/bedrift/minbedrift/beta/#/"') | (x.action == 'scroll on "Page: mobile-app.html"') | (x.action == 'scroll on "Page: https://www.telenor.no/bedrift/minbedrift/beta/mobile-app.html#/"'):
        return ('scroll_on_homepage')
    if ('on "Min Bedrift"' in x.action) | (x.action == 'click on "Page: https://www.telenor.no/bedrift/minbedrift/beta/"') | (x.action == 'click on "GÃ¥ tilbake til forsiden"'):
        return ('go_back_to_homepage')
    if 'on "TEXTAREA"' in x.action: # -- to be specified better if possible
        return('interact_with_pop_up_window')
    if ('on "Lokasjon"' in x.action):
        return ('click_on_locations_of subscriptions')
    if ('on "Pause"' in x.action):
        return ('click_on_pause')
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
    if (x.action == 'click on "arrow"'):
        return ('click_on_arrow')
    if ('on "Logg ut"' in x.action) | ('on "Log out"' in x.action):
        return ('click_log_out')
    if 'on "Meny"' in x.action:
        return ('click_on_menu')
    if 'on "Endre"' in x.action:
        return ('click_on_change')
    if ('on "Vilkår og betingelser"' in x.action) | (x.action == 'click on "VilkÃ¥r og betingelser"'):
        return ('click_on_terms_and_conditions')
    if ('Sjekk om din bedrift kan få fibe' in x.action) | (x.action =='click on ""Sjekk om din bedrift kan fÃ¥ fiber!""'):
        return ('click_whether_you_can_get_fiber')
    if 'click on "Ukjent"' in x.action:
        return ('click_on_unknown')
    if 'click on "Klarte ikke Ã¥ hente data"' in x.action:
        return ('click_on_failed_to_retrieve_data')
    if 'on "Oppsigelse"' in x.action:
        return ('click_on_resignation')
    if 'on "Personlige brukere"' in x.action:
        return ('click_on_personal_users')
    if x.action=='click on "selfOwner"':
        return ('click_on_self_owner')
    if x.action=='scroll on "Page : useragreement"':
        return ('scroll_on_page_user_agreement')
    if (x.action=='click on "Bruker"') | (x.action=='click on "User"'):
        return ('click_on_user')
    if (x.action=='click on "Adresse"') | (x.action=='click on "Adress"'):
        return ('click_on_address')
    if ('TO THE TOP' in x.action) | ('TIL TOPPEN' in x.action):
        return('go_back_to_top')
    if 'on "Lokasjoner"' in x.action:
        return ('click_on_menu_locations')
    if 'on "MBN Aktiv Bruker"' in x.action:
        return ('click_on_active_users')


    #if (x.action.str.actions('click on "edit-thin"')) or (x.action.str.actions('click on "Save"')) or (x.action.str.actions('click on "Lagre"')):
    #    return('completed_editing_subscription')

    # bugs/ feil
    if ('ng-valid-parse' in x.action) | ('ng-valid' in x.action) | ('ng-untouched' in x.action) | ('ng-dirty' in x.action):
        return ('empty_subselect_ignore')

    # Settings
    if ('on "NOR"' in x.action) | ('on "ENG"' in x.action):
        return ('change_language')
    if ('on "Innstillinger"' in x.action) |('on "Settings"' in x.action) :
        return('click_on_new_settings')

    # reports
    if ('scroll' in x.action) & ('reports' in x.action)|('rapporter' in x.action) | ('Rapporter' in x.action) | (x.action == 'click on "se rapport siden."'):
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
    if ('Januar' in x.action) | ('januar' in x.action) | ('Jan' in x.action) | ('jan' in x.action) | ('Feb' in x.action) | ('feb' in x.action) | ('mars' in x.action) | ('Mars' in x.action) | ('Mar' in x.action) | ('mar' in x.action) | ('april' in x.action) | ('April' in x.action)  | ('Apr' in x.action) | ('apr' in x.action) | ('mai' in x.action) | ('Mai' in x.action) | ('Jun' in x.action) | ('jun' in x.action) | ('Jul' in x.action) | ('jul' in x.action) | ('august' in x.action) | ('August' in x.action) | ('Aug' in x.action) | ('aug' in x.action) |  ('september' in x.action) | ('September' in x.action) | ('Sept' in x.action) | ('sept' in x.action) |  ('oktober' in x.action) | ('Oktober' in x.action) |  ('Okt' in x.action) | ('okt' in x.action)| ('november' in x.action) | ('November' in x.action) | ('Nov' in x.action) | ('nov' in x.action) |  ('desember' in x.action) | ('Desember' in x.action) | ('Des' in x.action) | ('des' in x.action):
        return('click_on_time_range')

    # dealer
    if 'Forhandlere' in x.action:
        return ('click_on_dealers')
    if 'Legg til forhandler' in x.action:
        return ('click_on_add_dealer')

    # invoice
    if 'on "Endre fakturareferanse"' in x.action:
        return('click_on_change_invoice_ref')
    if ('Fakturert' in x.action) | ('Invoiced' in x.action):
        return('click_on_billed')
    if ('on "ubetalte fakturaer"' in x.action) | ('on "Ubetalte fakturaer"' in x.action)| ('unpaid invoices' in x.action) | ('UtestÃ¥ende' in x.action) | (x.action == 'scroll on "Page: unpaidinvoices?type=mobile"'):
        return ('click_on_unpaid_invoices')
    if ('Faktureringsinformasjon' in x.action) | (x.action == 'click on "Faktureringsadresse"') | ('Faktura og betaling' in x.action):
        return ('click_on_billing_information')
    if (x.action == 'click on "Papirfaktura"'):
        return ('click_on_paper_invoice')
    if 'on "Betalt"' in x.action:
        return ('click_on_paid')
    if ('on "Se siste fakturaer"' in x.action) | ('on "See last invoices"' in x.action) | ('kostnader siden siste faktura funnet' in x.action) | (x.action =='click on "Se faktura"') | (x.action =='click on "faktura"') | (x.action == 'click on "Invoice number:"'):
        return ('see_invoice')
    if ('on "Fakturakonto -"' in x.action) | ('Fakturanummer' in x.action) | ('fakturakonto' in x.action) | ('click on "Fakturakonto"' in x.action) | ('click on "Fakturakontoer"' in x.action):
        return ('open_invoice_account')
    if ('on "Ny Fakturakonto"' in x.action) | ('on "New invoice account"' in x.action) | ('on "Opprett Fakturakonto"' in x.action) | ('on "Create invoice account"' in x.action):
        return ('click_on_new_invoice_account')
    if ('on "Bestill fakturakopi"' in x.action) | ('on "Bestill kopi"' in x.action) | ('on "Order copy"' in x.action) |(x.action == 'click on "Fakturakopi"'):
        return ('order_invoice_copy')
    if 'scroll on "Page: costs"' in x.action:
        return('scroll_on_page_costs')
    if 'Fakturakontroll' in x.action:
        return('scroll_on_fakturakontroll')
    if ('click on "ePostfaktura"' in x.action):
        return ('click_on_invoice_by_email')
    if ('on "Invoice inquiries"' in x.action) | ('on "Fakturahenvendelser"' in x.action) :
        return ('click_on_menu_invoice_enquiries')

    # administrators
    if ('Administratorer' in x.action) | ('Administrators' in x.action) | ('admin' in x.action) | (x.action == 'click on "Administrator og ansatt i selskapet"'):
        return('click_on_administrators')
    if 'scroll on "Page: admins"' in x.action:
        return('scroll_on_page_admins')
    if ('inaktive administrator' in x.action) | ('inactive administrator' in x.action):
        return('click_on_inactive_administrators')
    if 'on "Rolle og avdeling"' in x.action:
        return('click_on_role_department')
    if 'on "Gi samme tilgang som meg"' in x.action:
        return('click_on_give_the_same_access')
    if 'on "Kun lesetilgang"' in x.action:
        return('click_on_give_read_access')
    if ('on "Legg til administrator"' in x.action) | ('on "Ny administrator"' in x.action):
        return('click_on_add_admin')
    if ('on "Slett administrator"' in x.action) | ('on "Delete administrator"' in x.action):
        return('click_on_delete_admin')
    if 'on "Fjern tilgang"' in x.action:
        return('click_on_remove_access')
    if (x.action == 'click on "Fjern"') | (x.action == 'click on "Remove"'):
        return('click_on_remove')
    if 'on "Gi tilgang"' in x.action:
        return('click_on_give_access')
    if ('on "Ingen tilgang"' in x.action) | ('on "No access"' in x.action):
        return('click_on_no_access')
    # administrate companies
    if ('Selskap' in x.action) | ('on "Companies"' in x.action) | ('on "company"' in x.action) |('on "Company"' in x.action)| ('Companies' in x.action) | ('Selskaper' in x.action) | ('on "companies"' in x.action) |(x.action == 'scroll on "Page: companies"'):
        return('click_on_company')
    if 'on "egg til Selskaper"' in x.action:
        return('click_on_add_companies')
    if ('on "account"' in x.action) | ('Account' in x.action) | ('Kontoer' in x.action):
        return ('click_on_account')
    if x.action =='scroll on "Page: accounts"':
        return ('scroll_on_accounts_page')

    # product specific
    if (x.action == 'click on "Mobilt Bedriftsnett"') | ('on "mbn"' in x.action) | (x.action == 'click on "Mobile Bredbånd"') | ('on "Mobilt BredbÃ¥ndMobilt BredbÃ¥nd"' in x.action):
        return('click_on_MBN_sub')
    if (x.action == 'click on "Mobile"'):
        return('click_on_mobile_sub')
    if (x.action == 'click on "Mobilt Bredbånd 300GB"') | (x.action == 'click on "Mobilt Bredbånd 150GB"') | (x.action == 'click on "Mobilt Bredbånd 75GB"') | (x.action == 'click on "Mobilt Bredbånd 35GB"') | (x.action == 'click on "Mobilt Bredbånd 15GB"') | (x.action == 'click on "Mobilt Bredbånd 5GB"'):
        return('click_on_mobile_broadband_sub')
    if 'Bedrift Flyt' in x.action:
        return('click_on_bedrift_flyt_sub')
    if 'Bedrift' in x.action:
        return ('click_on_bedrift_sub')
    if (x.action == 'click on "Bedrift+ M"') | ('on "Bedrift XS"' in x.action) | (x.action == 'click on "Bedrift Fri+ 60GB"') | (x.action == 'click on "Bedrift Fri+ 40GB"') | (x.action == 'click on "Bedrift Fri+ 20GB"') | (x.action == 'click on "Bedrift Fri+ 4GB"') |(x.action == 'click on "Bedrift Fri+ 8GB"') | (x.action =='click on "Bedrift Fri+ 1GB"'):
        return('click_on_bedrift')
    if (x.action == 'click on "M2MMaskin til Maskin"') | (x.action == 'click on "M2M"') | (x.action == 'click on "M2M Go Norge"') | (x.action == 'click on "M2M Total"') | (x.action == 'click on "M2M Basis"'):
        return('click_on_m2m')
    if 'Bedrift Flyt' in x.action:
        return('click_on_bedrift_flyt')
    if x.action == 'click on "LTE-M"':
        return('click_on_lte')
    if 'Fastnett' in x.action:
        return('click_on_fixed')
    if 'Ingen datapakke inkludert' in x.action:
        return('click_on_no_data_incl')
    if ('on "data"' in x.action) | ('on "Datapakke"' in x.action) | ('datapakker' in x.action) | ('Data ekstrapakke' in x.action):
        return('click_on_data')
    if ('on "Mobilt BedriftsnettKonfigurerSe avtale"' in x.action):
        return('click_on_MBN_agreements')
    if 'MBN Aktiv Bruker39' in x.action:
        return ('click_on_mbnaktiv')
    if ('on "Surf' in x.action) | ('on "BTV Surf' in x.action) | ('on "DatapakkeSurf' in x.action):
        return ('click_on_surf_datapakke')
    # (additional) services
    if ('on "additional-services"' in x.action) | ('additionalServices' in x.action) | ('Tilleggstjenester' in x.action) :
        return ('adding_additional_services')
    if x.action == 'click on "Utforsk nye tjenester"':
        return ('explore_additional_services')
    if ('aktiv tjeneste' in x.action) | ('subscribed service' in x.action):
        return('click_on_active_cloud_services')
    if ('New cloud service' in x.action) | ('Ny skytjeneste' in x.action):
        return('click_on_order_new_cloud_service')
    if ('skytjeneste' in x.action) | ('Skytjeneste' in x.action) | ('click on "Cloud services"' in x.action) | ('on "cloud"' in x.action):
        return('click_on_cloud')
    if ('on "APIs' in x.action) | ('"APIer' in x.action):
        return ('click_on_APIs')
    if ('24SevenOffice' in x.action) | ('Active Directory Premium' in x.action) | ('Audio Conferencing' in x.action) | ('Domene' in x.action) |  ('Enterprise Mobility' in x.action) | \
            ('Exchange Advanced' in x.action) | ('Exchange Online' in x.action) | ('IBM' in x.action) | ('Microsoft' in x.action) |('Nettside' in x.action) | ('OneDrive' in x.action) |  \
            ('Phone System' in x.action) | ('Sharepoint' in x.action) |('Smart Dialog' in x.action)| ('Smartday Planner' in x.action)| ('Visio' in x.action) | ('Yammer' in x.action) | \
            ('Nettside' in x.action):
        return('click_on_cloud_product')

    # help
    if ('on "chat"' in x.action) | (x.action =='click on "Chat"'):
        return('start_chat')
    if ('on "Kontakt oss"' in x.action) | ('on "support"' in x.action) | ('Vi hjelper deg' in x.action) | ('hjelp' in x.action) | (x.action == 'click on "We will help you!"'):
        return('click_on_contact_us')
    if 'Feil' in x.action:
        return('click_on_error_details')
    if 'on "question"' in x.action:
        return('click_on_question')
    if 'information' in x.action:
        return('click_on_information')
    if 'Feedback' in x.action:
        return('click_on_feedback')

    # notifications
    if 'Varslinger' in x.action:
        return('click_on_notifications')
    if 'click on ""Merket du noe nytt med innloggingen?""' in x.action:
        return('click_on_did_you_notice_new_login')
    if ('Meldinger fra Telenor' in x.action) | ('unreadMessages' in x.action):
        return('click_on_inbox')
    if (x.action == 'scroll on "Page: messages"'):
        return('scroll_on_messages_page')
    if 'anbefaling' in x.action:
        return('click_on_recommendations')
    if (x.action=='click on "campaign-star"') | (x.action=='click on "Kampanjer"') | (x.action=='click on "Campaigns"'):
        return('click_on_campaigns')
    if "Rabatt" in x.action:
        return ('select_on_discounted_product')
    else:
        return('click_on_other')

# apply manual tagging function - REALLY SLOW!
# t['action_cleaned'] = t.apply(custom_function1, axis=1)
# check which events are untagged and most frequent (candidates for manual tagging)
# t[t['action_cleaned']=='click_on_other'].action.value_counts()
########### run in parallel
# 2.5 hours
t['action_cleaned'] = t.swifter.apply(tagging1, axis=1)
# inspect unlassified actions
t[t['action_cleaned']=='click_on_other'].action.value_counts()
###########################################################################
# identify and replace number entities
def entity_rec(i):
    if isinstance(i,float): # in case of nans (No longer needed if you deleted them already)
        pass
    else:
        i = nltk.word_tokenize(i)
        i = nltk.pos_tag(i)
        return i

def number_replace(row, output_format=None):
    d = entity_rec(row)
    # if Cardinal number, replace with "number"
    d1 = [('number', x[1]) if x[1] == 'CD' else x for x in d]
    if output_format == 'entities':  # return action split into components
        return(d1)
    else:  # returns replaced action with number replaced with "number"
        d2 = " ".join([ent[0] for ent in d1]).replace('``', '"').replace("''", '"')
        return(d2)

def name_replace(row, output_format=None):
    # needs to be run separately to avoide false replacemenets like Abbonement, subscription, SIM_card to "name"
    d = entity_rec(row)
    # if Cardinal number, replace with "number"
    d1 = [('name', x[1]) if x[1] == 'NNP' else x for x in d]
    if output_format == 'entities':  # return action split into components
        return(d1)
    else:  # returns replaced action with number replaced with "number"
        d2 = " ".join([ent[0] for ent in d1]).replace('``', '"').replace("''", '"')
        return(d2)

# 2,5 hours
t['action_numbers_repl'] = t.action.swifter.apply(number_replace)
###########################################################################
# manual tagging 2: extends "action_cleaned" to include actions that contained "numbers"
t[(t['action_cleaned']=='click_on_other')].action_numbers_repl.value_counts()

def tagging2(x):
    #### number entities tagging
    if (x.action_cleaned == 'click_on_other') & (('on " number number number "' in x.action_numbers_repl) | ('on " ( number ) "' in x.action_numbers_repl) | ('on " number "' in x.action_numbers_repl) | ('on " number number "' in x.action_numbers_repl) | ('on " number number number number "' in x.action_numbers_repl) | ('on " Page : number "' in x.action_numbers_repl)):
        return ('click_on_number_details')
    if (x.action_cleaned == 'click_on_other') & (('scroll on " Page : number "' in x.action_numbers_repl) | ('scroll on " number Abonnement "' in x.action_numbers_repl)):
        return ('scroll_on_number')
    if (x.action_cleaned == 'click_on_other') & (('on " Subscriptions ( number ) "' in x.action_numbers_repl) | ('on " Abonnement ( number ) "' in x.action_numbers_repl) | ('on " number abonnement "' in x.action_numbers_repl)):
        return ('click_on_subscription_detail')
    if (x.action_cleaned == 'click_on_other') & (('on " number Subscriptions "' in x.action_numbers_repl) | ('on " number number Subscriptions "' in x.action_numbers_repl) |('on " number Abonnement "' in x.action_numbers_repl) | ('on " number number Abonnement "' in x.action_numbers_repl)):
        return ('click_on_menu_subscriptions')
    if (x.action_cleaned == 'click_on_other') & (('on " SIM-kort ( number ) "' in x.action_numbers_repl) | ('on " SIM-card (  number  ) "' in x.action_numbers_repl)):
        return ('click_on_sim_card_details')
    if (x.action_cleaned == 'click_on_other') & ('on " Mobilt BredbÃ¥nd number "' in x.action_numbers_repl):
        return ('click_on_mbn_details')
    if (x.action_cleaned == 'click_on_other') & ('on " Sider ( number ) "' in x.action_numbers_repl):
        return ('click_on_pages_details')
    if (x.action_cleaned == 'click_on_other') & (('on " Administratorer ( number ) "' in x.action_numbers_repl) | ('on " Administrators (  number  ) "' in x.action_numbers_repl)):
        return ('click_on_administrators_details')
    if (x.action_cleaned == 'click_on_other') & (('on " number SIM-kort "' in x.action_numbers_repl) | ('on " number number SIM-kort "' in x.action_numbers_repl)| ('on " number SIM-card "' in x.action_numbers_repl) | ('on " number number SIM-card "' in x.action_numbers_repl)):
        return ('click_on_menu_sim_cards')
    if (x.action_cleaned == 'click_on_other') & (('on " number Accounts "' in x.action_numbers_repl) | ('on " number Fakturakontoer "' in x.action_numbers_repl) | ('on " number Fakturakonto "' in x.action_numbers_repl) | ('on " Fakturakonto - number "' in x.action_numbers_repl)):
        return ('click_on_menu_accounts')
    if (x.action_cleaned == 'click_on_other') & (('on " number Locations "' in x.action_numbers_repl) | ('on " number Lokasjoner "' in x.action_numbers_repl)):
        return ('click_on_menu_locations')
    if (x.action_cleaned == 'click_on_other') & ('keypress < number > on " simNumber "' in x.action_numbers_repl):
        return ('click_on_sim_number_details')
    if (x.action_cleaned == 'click_on_other') & (('click on " Se number avtaler "' in x.action_numbers_repl) | ('click on " Se number avtaleprodukter "' in x.action_numbers_repl) | ('" number Avtaleprodukt "' in x.action_numbers_repl)):
        return ('click_on_agreement_detail')
    if (x.action_cleaned == 'click_on_other') & ('on " number Forhandler "' in x.action_numbers_repl):
        return ('click_on_dealer_list')
    if (x.action_cleaned == 'click_on_other') & ('on " Sikkerhet ( number ) "' in x.action_numbers_repl):
        return ('click_on_safety_products')
    if (x.action_cleaned == 'click_on_other') & ('on " number selskaper "' in x.action_numbers_repl) | ('on " number selskap "' in x.action_numbers_repl):
        return ('click_on_companies_with_product')
    if (x.action_cleaned == 'click_on_other') & (x.action_numbers_repl == 'keypress < number > on " Page : https : //www.telenor.no/bedrift/minbedrift/beta/ "'):
        return ('click_on_homepage')
    if (x.action_cleaned == 'click_on_other') & (('on " number , - "' in x.action_numbers_repl) | ('on " number , - /mnd "' in x.action_numbers_repl)):
        return ('click_on_price')
    if (x.action_cleaned == 'click_on_other') & (('number Enhet' in x.action_numbers_repl) ):
        return ('click_on_enhet_details')
    if (x.action_cleaned == 'click_on_other') & ('click on " number Brukere "' in x.action_numbers_repl) :
        return ('click_on_user_details')
    if (x.action_cleaned == 'click_on_other') & ('on " Du har number kampanje "' in x.action_numbers_repl) :
        return ('click_on_campaign_details')
    else:
        return(x.action_cleaned)

t['action_cleaned'] =  t.swifter.apply(tagging2, axis=1)

t[(t['action_cleaned']=='click_on_other')].action_numbers_repl.value_counts()

###########################################################################
t['action_numbers_repl'] = t.action.swifter.apply(name_replace)

def tagging3(x):
    ### name entities tagging
    if (x.action_cleaned == 'click_on_other') & (('on " name , name "' in x.action_numbers_repl) | ('on " name , name name "' in x.action_numbers_repl) | ('on " name name name' in x.action_numbers_repl) |
                                                 ('on " name As "' in x.action_numbers_repl) | ('on " name name As "' in x.action_numbers_repl) | ('on " name AS "' in x.action_numbers_repl) | ('on " name name , name "' in x.action_numbers_repl) |
                                                 ('on " name name AS "' in x.action_numbers_repl) |('click on " name name name AS "' in x.action_numbers_repl) |('on " name name name "' in x.action_numbers_repl) | ('on " name name name name "' in x.action_numbers_repl) |
                                                 ('on " name name "' in x.action_numbers_repl) | ('on " name , name name name "' in x.action_numbers_repl) | ('on " name name , name name "' in x.action_numbers_repl)|
                                                 ('on " name "' in x.action_numbers_repl) | ('on " name name & name name "' in x.action_numbers_repl) | ('on " name & name name "' in x.action_numbers_repl)) :
        return ('click_on_name')
    else:
        return(x.action_cleaned)

t['action_cleaned'] =  t.swifter.apply(tagging3, axis=1)
###########################################################################
t[(t['action_cleaned']=='click_on_other')].action.value_counts()
t[(t['action_cleaned']=='click_on_other')].action_numbers_repl.value_counts()

# check how much is tagged
t[(t['action_cleaned']!='click_on_other')].shape[0]/t.shape[0]

